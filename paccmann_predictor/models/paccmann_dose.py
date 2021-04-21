import logging
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from ..utils.interpret import monte_carlo_dropout, test_time_augmentation
from ..utils.layers import dense_layer
from . import PaccMannV2

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PaccMannDose(PaccMannV2):
    """PaccMann Multiscale Convolutional Attentive (MCA) model to predict cell
    viability using drug concentration as an additional input stream.
    """

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense encoder.
                TODO params should become actual arguments (use **params).

        Items in params:
            smiles_embedding_size (int): dimension of tokens' embedding.
            smiles_vocabulary_size (int): size of the tokens vocabulary.

            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Defaults to 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Defaults to True.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Defaults to 0.5.
            filters (list[int], optional): Numbers of filters to learn per
                convolutional layer. Defaults to [64, 64, 64].
            kernel_sizes (list[list[int]], optional): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['smiles_embedding_size']],
                    [5, params['smiles_embedding_size']],
                    [11, params['smiles_embedding_size']]
                ]
                NOTE: The kernel sizes should match the dimensionality of the
                smiles_embedding_size, so if the latter is 8, the images are
                t x 8, then treat the 8 embedding dimensions like channels
                in an RGB image.
            molecule_heads (list[int], optional): Amount of attentive multiheads
                per SMILES embedding. Should have len(filters)+1.
                Defaults to [4, 4, 4, 4].
            stacked_dense_hidden_sizes (list[int], optional): Sizes of the
                hidden dense layers. Defaults to [1024, 512].
            smiles_attention_size (int, optional): size of the attentive layer
                for the smiles sequence. Defaults to 64.
    """

        super(PaccMannDose, self).__init__(params, *args, **kwargs)

        # One extra neuron for concentration.
        self.hidden_sizes[0] += 1
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dense_{}'.format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get('batch_norm', True)
                        ).to(self.device)
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

    def forward(self, smiles, gep, dose, confidence=False):
        """Forward pass through the MCA.

        Args:
            smiles (torch.Tensor): of type int and shape `[bs, seq_length]`.
            gep (torch.Tensor): of shape `[bs, num_genes]`.
            dose (torch.Tensor): of shape `[bs, 1]`.
            confidence (bool, optional) whether the confidence estimates are
                performed.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is cell viability tensor of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """
        geps = torch.unsqueeze(gep, dim=-1)
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # SMILES Convolutions. Unsqueeze has shape bs x 1 x T x H.
        encoded_smiles = [embedded_smiles] + [
            self.convolutional_layers[ind]
            (torch.unsqueeze(embedded_smiles, 1)).permute(0, 2, 1)
            for ind in range(len(self.convolutional_layers))
        ]

        # Molecule context attention
        encodings, smiles_alphas, gene_alphas = [], [], []
        for layer in range(len(self.molecule_heads)):
            for head in range(self.molecule_heads[layer]):

                ind = self.molecule_heads[0] * layer + head
                e, a = self.molecule_attention_layers[ind](
                    encoded_smiles[layer], geps
                )
                encodings.append(e)
                smiles_alphas.append(a)

        # Gene context attention
        for layer in range(len(self.gene_heads)):
            for head in range(self.gene_heads[layer]):
                ind = self.gene_heads[0] * layer + head

                e, a = self.gene_attention_layers[ind](
                    geps, encoded_smiles[layer], average_seq=False
                )
                encodings.append(e)
                gene_alphas.append(a)

        encodings = torch.cat(encodings, dim=1)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get(
            'batch_norm', False
        ) else encodings

        inputs = torch.cat([inputs, dose], dim=1)
        # Stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {}

        if not self.training:
            # The below is to ease postprocessing
            smiles_attention = torch.cat(
                [torch.unsqueeze(p, -1) for p in smiles_alphas], dim=-1
            )
            gene_attention = torch.cat(
                [torch.unsqueeze(p, -1) for p in gene_alphas], dim=-1
            )
            prediction_dict.update({
                'gene_attention': gene_attention,
                'smiles_attention': smiles_attention,
                'viability': predictions
            })  # yapf: disable

            if confidence:
                augmenter = AugmentTensor(self.smiles_language)
                epi_conf, epi_pred = monte_carlo_dropout(
                    self,
                    regime='tensors',
                    tensors=(smiles, gep, dose),
                    repetitions=5
                )
                ale_conf, ale_pred = test_time_augmentation(
                    self,
                    regime='tensors',
                    tensors=(smiles, gep, dose),
                    repetitions=5,
                    augmenter=augmenter,
                    tensors_to_augment=0
                )

                prediction_dict.update({
                    'epistemic_confidence': epi_conf,
                    'epistemic_predictions': epi_pred,
                    'aleatoric_confidence': ale_conf,
                    'aleatoric_predictions': ale_pred
                })  # yapf: disable

        elif confidence:
            logger.info('Using confidence in training mode is not supported.')

        return predictions, prediction_dict
