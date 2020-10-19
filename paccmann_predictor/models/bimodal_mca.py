from collections import OrderedDict

import pytoda
import torch
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from ..utils.hyperparams import ACTIVATION_FN_FACTORY, LOSS_FN_FACTORY
from ..utils.interpret import monte_carlo_dropout, test_time_augmentation
from ..utils.layers import (
    ContextAttentionLayer,
    convolutional_layer,
    dense_layer,
)
from ..utils.utils import get_device


class BimodalMCA(nn.Module):
    """Bimodal Multiscale Convolutional Attentive Encoder.

    This is based on the MCA model as presented in the publication in
    Molecular Pharmaceutics:
        https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520.
    """

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense encoder.
                TODO params should become actual arguments (use **params).

        Required items in params:
            smiles_padding_length (int): dimension of tokens' embedding.
            smiles_vocabulary_size (int): size of the tokens vocabulary.
            protein_padding_length (int): dimension of tokens' embedding.
            protein_vocabulary_size (int): size of the tokens vocabulary.
        Optional items in params:
            activation_fn (string): Activation function used in all ayers for
                specification in ACTIVATION_FN_FACTORY. Defaults to 'relu'.
            batch_norm (bool): Whether batch normalization is applied. Defaults
                to True.
            dropout (float): Dropout probability in all except context
                attention layer. Defaults to 0.5.
            smiles_embedding_size (int): Embedding dimensionality, default: 32
            protein_embedding_size (int): Embedding dimensionality, default: 8
            smiles_filters (list[int]): Numbers of filters to learn per
                convolutional layer. Defaults to [32, 32, 32].
            protein_filters (list[int]): Numbers of filters to learn per
                convolutional layer. Defaults to [32, 32, 32].
            smiles_kernel_sizes (list[list[int]]): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['smiles_embedding_size']],
                    [5, params['smiles_embedding_size']],
                    [11, params['smiles_embedding_size']]
                ]
            protein_kernel_sizes (list[list[int]]): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['protein_embedding_size']],
                    [11, params['protein_embedding_size']],
                    [25, params['protein_embedding_size']]
                ]
                NOTE: The kernel sizes should match the dimensionality of the
                smiles_embedding_size, so if the latter is 8, the images are
                t x 8, then treat the 8 embedding dimensions like channels
                in an RGB image.
            smiles_attention_size (int): size of the attentive layer for the
                smiles sequence. Defaults to 16.
            protein_attention_size (int): size of the attentive layer for the
                protein sequence. Defaults to 16.
            dense_hidden_sizes (list[int]): Sizes of the hidden dense layers.
                Defaults to [20].
            final_activation: (bool): Whether a (sigmoid) activation function
                is used in the final layer. Defaults to False.
        """

        super(BimodalMCA, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.smiles_padding_length = params['smiles_padding_length']
        self.protein_padding_length = params['protein_padding_length']

        self.loss_fn = LOSS_FN_FACTORY[
            params.get('loss_fn', 'binary_cross_entropy')
        ]  # yapf: disable

        # Hyperparameter
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')
        ]  # yapf: disable
        self.dropout = params.get('dropout', 0.5)
        self.use_batch_norm = params.get('batch_norm', True)
        self.temperature = params.get('temperature', 1.0)
        self.smiles_embedding_size = params.get('smiles_embedding_size', 32)
        self.protein_embedding_size = params.get('protein_embedding_size', 8)
        self.smiles_filters = params.get('smiles_filters', [32, 32, 32])
        self.protein_filters = params.get('protein_filters', [32, 32, 32])

        self.smiles_kernel_sizes = params.get(
            'smiles_kernel_sizes',
            [
                [3, self.smiles_embedding_size],
                [5, self.smiles_embedding_size],
                [11, self.smiles_embedding_size],
            ],
        )
        self.protein_kernel_sizes = params.get(
            'protein_kernel_sizes',
            [
                [3, self.protein_embedding_size],
                [11, self.protein_embedding_size],
                [25, self.protein_embedding_size],
            ],
        )

        self.smiles_attention_size = params.get('smiles_attention_size', 16)
        self.protein_attention_size = params.get('protein_attention_size', 16)

        self.smiles_hidden_sizes = [
            self.smiles_embedding_size
        ] + self.smiles_filters
        self.protein_hidden_sizes = [
            self.protein_embedding_size
        ] + self.protein_filters
        self.hidden_sizes = [
            self.smiles_embedding_size
            + sum(self.smiles_filters)
            + self.protein_embedding_size
            + sum(self.protein_filters)
        ] + params.get('dense_hidden_sizes', [20])
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])

        # Sanity checking of model sizes
        if len(self.smiles_filters) != len(self.smiles_kernel_sizes):
            raise ValueError(
                'Length of SMILES filter and kernel size lists do not match.'
            )
        if len(self.protein_filters) != len(self.protein_kernel_sizes):
            raise ValueError(
                'Length of Protein filter and kernel size lists do not match.'
            )
        if len(self.smiles_filters) != len(self.protein_filters):
            raise ValueError(
                'Length of smiles_filters and protein_filters array must match'
                f', found smiles_filters: {len(self.smiles_filters)} and '
                f'protein_filters: {len(self.protein_filters)}.'
            )
        """ Construct model  """
        # Embeddings
        self.smiles_embedding = nn.Embedding(
            self.params['smiles_vocabulary_size'],
            self.smiles_embedding_size,
            scale_grad_by_freq=params.get('embed_scale_grad', False),
        )
        self.protein_embedding = nn.Embedding(
            self.params['protein_vocabulary_size'],
            self.protein_embedding_size,
            scale_grad_by_freq=params.get('embed_scale_grad', False),
        )

        # Convolutions
        # TODO: Use nn.ModuleDict instead of the nn.Seq/OrderedDict
        self.smiles_convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'smiles_convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.smiles_filters, self.smiles_kernel_sizes)
                    )
                ]
            )
        )  # yapf: disable

        self.protein_convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'protein_convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.protein_filters, self.protein_kernel_sizes)
                    )
                ]
            )
        )  # yapf: disable

        # Context attention
        self.context_attention_smiles_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'context_attention_smiles_{layer}',
                        ContextAttentionLayer(
                            self.smiles_hidden_sizes[layer],
                            self.params['smiles_padding_length'],
                            self.protein_hidden_sizes[layer],
                            context_sequence_length=(
                                self.protein_padding_length
                            ),
                            attention_size=self.smiles_attention_size,
                            individual_nonlinearity=params.get(
                                'context_nonlinearity', nn.Sequential()
                            ),
                            temperature=self.temperature,
                        ),
                    )
                    for layer in range(len(self.smiles_filters) + 1)
                ]
            )
        )

        self.context_attention_protein_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'context_attention_protein_{layer}',
                        ContextAttentionLayer(
                            self.protein_hidden_sizes[layer],
                            self.params['protein_padding_length'],
                            self.smiles_hidden_sizes[layer],
                            context_sequence_length=self.smiles_padding_length,
                            attention_size=self.protein_attention_size,
                            individual_nonlinearity=params.get(
                                'context_nonlinearity', nn.Sequential()
                            ),
                            temperature=self.temperature,
                        ),
                    )
                    for layer in range(len(self.protein_filters) + 1)
                ]
            )
        )

        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'dense_{ind}',
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = nn.Linear(self.hidden_sizes[-1], 1)
        if params.get('final_activation', True):
            self.final_dense = nn.Sequential(
                self.final_dense, ACTIVATION_FN_FACTORY['sigmoid']
            )

    def forward(self, smiles, proteins, confidence=False):
        """Forward pass through the biomodal MCA.

        Args:
            smiles (torch.Tensor): of type int and shape
                `[bs, smiles_padding_length]`.
            proteins (torch.Tensor): of type int and shape
                `[bs, protein_padding_length]`.
            confidence (bool, optional) whether the confidence estimates are
                performed.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """

        # Embedding
        embedded_smiles = self.smiles_embedding(smiles.to(torch.int64))
        embedded_protein = self.protein_embedding(proteins.to(torch.int64))

        # Convolutions
        encoded_smiles = [embedded_smiles] + [
            layer(torch.unsqueeze(embedded_smiles, 1)).permute(0, 2, 1)
            for layer in self.smiles_convolutional_layers
        ]
        encoded_protein = [embedded_protein] + [
            layer(torch.unsqueeze(embedded_protein, 1)).permute(0, 2, 1)
            for layer in self.protein_convolutional_layers
        ]

        # Context attention on SMILES
        smiles_encodings, smiles_alphas = zip(
            *[
                layer(reference, context)
                for layer, reference, context in zip(
                    self.context_attention_smiles_layers,
                    encoded_smiles,
                    encoded_protein,
                )
            ]
        )

        # Context attention on Protein
        protein_encodings, protein_alphas = zip(
            *[
                layer(reference, context)
                for layer, reference, context in zip(
                    self.context_attention_protein_layers,
                    encoded_protein,
                    encoded_smiles,
                )
            ]
        )

        # Concatenate all encodings
        encodings = torch.cat(
            [
                torch.cat(smiles_encodings, dim=1),
                torch.cat(protein_encodings, dim=1),
            ],
            dim=1,
        )

        # Apply batch normalization if specified
        out = self.batch_norm(encodings) if self.use_batch_norm else encodings

        # Stack dense layers
        for dl in self.dense_layers:
            out = dl(out)
        predictions = self.final_dense(out)

        prediction_dict = {}
        if not self.training:
            # The below is to ease postprocessing
            smiles_attention_weights = torch.mean(
                torch.cat(
                    [torch.unsqueeze(p, -1) for p in smiles_alphas], dim=-1
                ),
                dim=-1,
            )
            protein_attention_weights = torch.mean(
                torch.cat(
                    [torch.unsqueeze(p, -1) for p in protein_alphas], dim=-1
                ),
                dim=-1,
            )
            prediction_dict.update(
                {
                    'smiles_attention': smiles_attention_weights,
                    'protein_attention': protein_attention_weights,
                }
            )  # yapf: disable

            if confidence:
                augmenter = AugmentTensor(self.smiles_language)
                epistemic_conf = monte_carlo_dropout(
                    self,
                    regime='tensors',
                    tensors=(smiles, proteins),
                    repetitions=5,
                )
                aleatoric_conf = test_time_augmentation(
                    self,
                    regime='tensors',
                    tensors=(smiles, proteins),
                    repetitions=5,
                    augmenter=augmenter,
                    tensors_to_augment=0,
                )

                prediction_dict.update(
                    {
                        'epistemic_confidence': epistemic_conf,
                        'aleatoric_confidence': aleatoric_conf,
                    }
                )  # yapf: disable

        return predictions, prediction_dict

    def loss(self, yhat, y):
        return self.loss_fn(yhat, y)

    def _associate_language(self, language):
        """
        Bind a SMILES or Protein language object to the model.
        Is only used inside the confidence estimation.

        Arguments:
            language {Union[
                pytoda.smiles.smiles_language.SMILESLanguage,
                pytoda.proteins.protein_langauge.ProteinLanguage
            ]} -- [A SMILES or Protein language object]

        Raises:
            TypeError:
        """
        if isinstance(language, pytoda.smiles.smiles_language.SMILESLanguage):
            self.smiles_language = language

        elif isinstance(
            language, pytoda.proteins.protein_language.ProteinLanguage
        ):
            self.protein_language = language
        else:
            raise TypeError(
                'Please insert a smiles language (object of type '
                'pytoda.smiles.smiles_language.SMILESLanguage or '
                'pytoda.proteins.protein_language.ProteinLanguage). Given was '
                f'{type(language)}'
            )

    def load(self, path, *args, **kwargs):
        """Load model from path."""
        weights = torch.load(path, *args, **kwargs)
        self.load_state_dict(weights)

    def save(self, path, *args, **kwargs):
        """Save model to path."""
        torch.save(self.state_dict(), path, *args, **kwargs)
