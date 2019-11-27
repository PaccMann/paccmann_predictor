from collections import OrderedDict

import torch
import torch.nn as nn

from ..utils.hyperparams import ACTIVATION_FN_FACTORY, LOSS_FN_FACTORY
from ..utils.layers import (
    alpha_projection, convolutional_layer, dense_attention_layer, dense_layer,
    gene_projection, smiles_projection
)
from ..utils.utils import get_device


class MCA(nn.Module):
    """Multiscale Convolutional Attentive Encoder.

    This is the MCA model as presented in the authors publication in
    Molecular Pharmaceutics https://arxiv.org/abs/1904.11223.
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
            multiheads (list[int], optional): Amount of attentive multiheads
                per SMILES embedding. Should have len(filters)+1.
                Defaults to [4, 4, 4, 4].
            stacked_dense_hidden_sizes (list[int], optional): Sizes of the
                hidden dense layers. Defaults to [1024, 512].
            smiles_attention_size (int, optional): size of the attentive layer
                for the smiles sequence. Defaults to 64.

        Example params:
        ```
        {
            "smiles_attention_size": 8,
            "smiles_vocabulary_size": 28,
            "smiles_embedding_size": 8,
            "filters": [128, 128],
            "kernel_sizes": [[3, 8], [5, 8]],
            "multiheads":[32, 32, 32]
            "stacked_dense_hidden_sizes": [512, 64, 16]
        }
        ```
    """

        super(MCA, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]

        # Model inputs
        self.number_of_genes = params.get('number_of_genes', 2128)
        self.smiles_attention_size = params.get('smiles_attention_size', 64)

        # Model architecture (hyperparameter)
        self.multiheads = params.get('multiheads', [4, 4, 4, 4])
        self.filters = params.get('filters', [64, 64, 64])
        self.hidden_sizes = (
            [
                self.multiheads[0] * params['smiles_embedding_size'] + sum(
                    [h * f for h, f in zip(self.multiheads[1:], self.filters)]
                )
            ] + params.get('stacked_dense_hidden_sizes', [1024, 512])
        )

        if params.get('gene_to_dense', False):  # Optional skip connection
            self.hidden_sizes[0] += self.number_of_genes
        self.dropout = params.get('dropout', 0.5)
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')]
        self.kernel_sizes = params.get(
            'kernel_sizes', [
                [3, params['smiles_embedding_size']],
                [5, params['smiles_embedding_size']],
                [11, params['smiles_embedding_size']]
            ]
        )
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError(
                'Length of filter and kernel size lists do not match.'
            )
        if len(self.filters) + 1 != len(self.multiheads):
            raise ValueError(
                'Length of filter and multihead lists do not match'
            )

        # Build the model
        self.smiles_embedding = nn.Embedding(
            self.params['smiles_vocabulary_size'],
            self.params['smiles_embedding_size'],
            scale_grad_by_freq=params.get('embed_scale_grad', False)
        )
        self.gene_attention_layer = dense_attention_layer(
            self.number_of_genes
        ).to(self.device)

        self.convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            batch_norm=params.get('batch_norm', False),
                            dropout=self.dropout
                        ).to(self.device)
                    ) for index, (num_kernel, kernel_size) in
                    enumerate(zip(self.filters, self.kernel_sizes))
                ]
            )
        )

        smiles_hidden_sizes = [params['smiles_embedding_size']] + self.filters

        # Defines contextual attention mechanism
        self.gene_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'gene_projection_{self.multiheads[0]*layer+index}',
                        gene_projection(
                            self.number_of_genes, self.smiles_attention_size
                        )
                    ) for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )
        self.smiles_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'smiles_projection_{self.multiheads[0]*layer+index}',
                        smiles_projection(
                            smiles_hidden_sizes[layer],
                            self.smiles_attention_size
                        )
                    ) for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )
        self.alpha_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'alpha_projection_{self.multiheads[0]*layer+index}',
                        alpha_projection(self.smiles_attention_size)
                    ) for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )
        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
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

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get('final_activation', False) else nn.Sequential(
                OrderedDict(
                    [
                        ('projection', nn.Linear(self.hidden_sizes[-1], 1)),
                        ('sigmoidal', ACTIVATION_FN_FACTORY['sigmoid'])
                    ]
                )
            )
        )

    def forward(self, smiles, gep):
        """Forward pass through the MCA.

        Args:
            smiles (torch.Tensor): of type int and shape `[bs, seq_length]`.
            gep (torch.Tensor): of shape `[bs, num_genes]`.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """

        embedded_smiles = self.smiles_embedding(smiles.to(dtype=torch.int64))

        # Gene attention weights
        gene_alphas = self.gene_attention_layer(gep)

        # Filter the gene expression with the weights.
        encoded_genes = gene_alphas * gep

        # NOTE: SMILES Convolutions. Unsqueeze has shape bs x 1 x T x H.
        encoded_smiles = [embedded_smiles] + [
            self.convolutional_layers[ind]
            (torch.unsqueeze(embedded_smiles, 1)).permute(0, 2, 1)
            for ind in range(len(self.convolutional_layers))
        ]

        # NOTE: SMILES Attention mechanism
        smiles_alphas, encodings = [], []
        for layer in range(len(self.multiheads)):
            for head in range(self.multiheads[layer]):
                ind = self.multiheads[0] * layer + head
                gene_context = self.gene_projections[ind](encoded_genes)
                smiles_context = self.smiles_projections[ind](
                    encoded_smiles[layer]
                )

                smiles_alphas.append(
                    self.alpha_projections[ind](
                        torch.tanh(gene_context + smiles_context)
                    )
                )
                # Sequence is always reduced.
                encodings.append(
                    torch.sum(
                        encoded_smiles[layer] *
                        torch.unsqueeze(smiles_alphas[-1], -1), 1
                    )
                )
        encodings = torch.cat(encodings, dim=1)
        if self.params.get('gene_to_dense', False):
            encodings = torch.cat([encodings, gep], dim=1)

        # Apply batch normalization if specified
        inputs = self.batch_norm(encodings) if self.params.get(
            'batch_norm', False
        ) else encodings
        # NOTE: stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {
            'gene_attention': gene_alphas,
            'smiles_attention': smiles_alphas,
            'IC50': predictions,
        }
        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
