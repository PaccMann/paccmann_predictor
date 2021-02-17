from collections import OrderedDict
import pickle

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
            ligand_padding_length (int): dimension of tokens' embedding.
            ligand_vocabulary_size (int): size of the tokens vocabulary.
            receptor_padding_length (int): dimension of tokens' embedding.
            receptor_vocabulary_size (int): size of the tokens vocabulary.
        Optional items in params:
            activation_fn (str): Activation function used in all ayers for
                specification in ACTIVATION_FN_FACTORY. Defaults to 'relu'.
            batch_norm (bool): Whether batch normalization is applied. Defaults
                to True.
            dropout (float): Dropout probability in all except context
                attention layer. Defaults to 0.5.
            ligand_embedding (str): Way to numberically embed ligand sequence.
                Options: 'predefined' (sequence is already embedded using
                predefined token representations like BLOSUM matrix),
                'one-hot', 'pretrained' (loads embedding from ligand_embedding
                path) or 'learned (model learns an embedding from data).
                Defaults to 'learned'.
            ligand_embedding_path (str): Path where pretrained embedding
                weights are stored. Needed if ligand_embedding is 'pretrained'.
            receptor_embedding (str): Way to numberically embed receptor sequence.
                Options: 'predefined' (sequence is already embedded using
                predefined token representations like BLOSUM matrix),
                'one-hot', 'pretrained' (loads embedding from receptor_embedding
                path) or 'learned (model learns an embedding from data).
                Defaults to 'learned'.
            receptor_embedding_path (str): Path where pretrained embedding
                weights are stored. Needed if receptor_embedding is 'pretrained'.
            ligand_embedding_size (int): Embedding dimensionality, default: 32
            receptor_embedding_size (int): Embedding dimensionality, default: 8
            ligand_filters (list[int]): Numbers of filters to learn per
                convolutional layer. Defaults to [32, 32, 32].
            receptor_filters (list[int]): Numbers of filters to learn per
                convolutional layer. Defaults to [32, 32, 32].
            ligand_kernel_sizes (list[list[int]]): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['ligand_embedding_size']],
                    [5, params['ligand_embedding_size']],
                    [11, params['ligand_embedding_size']]
                ]
            receptor_kernel_sizes (list[list[int]]): Sizes of kernels per
                convolutional layer. Defaults to  [
                    [3, params['receptor_embedding_size']],
                    [11, params['receptor_embedding_size']],
                    [25, params['receptor_embedding_size']]
                ]
                NOTE: The kernel sizes should match the dimensionality of the
                ligand_embedding_size, so if the latter is 8, the images are
                t x 8, then treat the 8 embedding dimensions like channels
                in an RGB image.
            ligand_attention_size (int): size of the attentive layer for the
                ligand sequence. Defaults to 16.
            receptor_attention_size (int): size of the attentive layer for the
                receptor sequence. Defaults to 16.
            dense_hidden_sizes (list[int]): Sizes of the hidden dense layers.
                Defaults to [20].
            final_activation: (bool): Whether a (sigmoid) activation function
                is used in the final layer. Defaults to False.
        """

        super(BimodalMCA, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.ligand_padding_length = params['ligand_padding_length']
        self.receptor_padding_length = params['receptor_padding_length']

        self.loss_fn = LOSS_FN_FACTORY[
            params.get('loss_fn', 'binary_cross_entropy')
        ]  # yapf: disable
        self.ligand_embedding_type = params.get('ligand_embedding', 'learned')
        self.receptor_embedding_type = params.get(
            'receptor_embedding', 'learned'
        )

        # Hyperparameter
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')
        ]  # yapf: disable
        self.dropout = params.get('dropout', 0.5)
        self.use_batch_norm = params.get('batch_norm', True)
        self.temperature = params.get('temperature', 1.0)
        self.ligand_filters = params.get('ligand_filters', [32, 32, 32])
        self.receptor_filters = params.get('receptor_filters', [32, 32, 32])

        # set embedding_size to vocabulary_size if one_hot encoding is chosen
        if params.get('ligand_embedding', 'learned') == 'one_hot':
            self.ligand_embedding_size = params.get(
                'ligand_vocabulary_size', 32
            )
        else:
            self.ligand_embedding_size = params.get(
                'ligand_embedding_size', 32
            )
        if params.get('receptor_embedding', 'learned') == 'one_hot':
            self.receptor_embedding_size = params.get(
                'receptor_vocabulary_size', 35
            )
        else:
            self.receptor_embedding_size = params.get(
                'receptor_embedding_size', 35
            )

        self.ligand_kernel_sizes = params.get(
            'ligand_kernel_sizes',
            [
                [3, self.ligand_embedding_size],
                [5, self.ligand_embedding_size],
                [11, self.ligand_embedding_size],
            ],
        )
        self.receptor_kernel_sizes = params.get(
            'receptor_kernel_sizes',
            [
                [3, self.receptor_embedding_size],
                [11, self.receptor_embedding_size],
                [25, self.receptor_embedding_size],
            ],
        )

        self.ligand_attention_size = params.get('ligand_attention_size', 16)
        self.receptor_attention_size = params.get(
            'receptor_attention_size', 16
        )

        self.ligand_hidden_sizes = [
            self.ligand_embedding_size
        ] + self.ligand_filters
        self.receptor_hidden_sizes = [
            self.receptor_embedding_size
        ] + self.receptor_filters
        self.hidden_sizes = [
            self.ligand_embedding_size + sum(self.ligand_filters) +
            self.receptor_embedding_size + sum(self.receptor_filters)
        ] + params.get('dense_hidden_sizes', [20])
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])

        # Sanity checking of model sizes
        if len(self.ligand_filters) != len(self.ligand_kernel_sizes):
            raise ValueError(
                'Length of ligand filter and kernel size lists do not match.'
            )
        if len(self.receptor_filters) != len(self.receptor_kernel_sizes):
            raise ValueError(
                'Length of receptor filter and kernel size lists do not match.'
            )
        if len(self.ligand_filters) != len(self.receptor_filters):
            raise ValueError(
                'Length of ligand_filters and receptor_filters array must match'
                f', found ligand_filters: {len(self.ligand_filters)} and '
                f'receptor_filters: {len(self.receptor_filters)}.'
            )
        """ Construct model  """
        # Embeddings
        if params.get('ligand_embedding', 'learned') == 'pretrained':
            # Load the pretrained embeddings
            try:
                with open(params['ligand_embedding_path'], 'rb') as f:
                    embeddings = pickle.load(f)
            except KeyError:
                raise KeyError('Path for ligand embeddings missing in params.')

            # Plug into layer
            self.ligand_embedding = nn.Embedding(
                embeddings.shape[0], embeddings.shape[1]
            )
            self.ligand_embedding.load_state_dict(
                {'weight': torch.Tensor(embeddings)}
            )
            if params.get('fix_ligand_embeddings', True):
                self.ligand_embedding.weight.requires_grad = False

        elif params.get('ligand_embedding', 'learned') == 'one_hot':
            self.ligand_embedding = nn.Embedding(
                self.params['ligand_vocabulary_size'],
                self.params['ligand_vocabulary_size'],
            )
            # Plug in one hot-vectors and freeze weights
            self.ligand_embedding.load_state_dict(
                {
                    'weight':
                        torch.nn.functional.one_hot(
                            torch.arange(
                                self.params['ligand_vocabulary_size']
                            )
                        )
                }
            )
            self.ligand_embedding.weight.requires_grad = False

        elif params.get('ligand_embedding', 'learned') == 'learned':
            self.ligand_embedding = nn.Embedding(
                self.params['ligand_vocabulary_size'],
                self.ligand_embedding_size,
                scale_grad_by_freq=params.get('embed_scale_grad', False)
            )
        else:
            assert params.get(
                'ligand_embedding', 'learned'
            ) == 'predefined', 'Choose either pretrained, one_hot, predefined \
             or learned as ligand_embedding. Defaults to learned'

        if params.get('receptor_embedding', 'learned') == 'pretrained':
            # Load the pretrained embeddings
            try:
                with open(params['receptor_embedding_path'], 'rb') as f:
                    embeddings = pickle.load(f)
            except KeyError:
                raise KeyError(
                    'Path for receptor embeddings missing in params.'
                )

            # Plug into layer
            self.receptor_embedding = nn.Embedding(
                embeddings.shape[0], embeddings.shape[1]
            )
            self.receptor_embedding.load_state_dict(
                {'weight': torch.Tensor(embeddings)}
            )
            if params.get('fix_receptor_embeddings', True):
                self.receptor_embedding.weight.requires_grad = False

        elif params.get('receptor_embedding', 'learned') == 'one_hot':
            self.receptor_embedding = nn.Embedding(
                self.params['receptor_vocabulary_size'],
                self.params['receptor_vocabulary_size'],
            )
            # Plug in one hot-vectors and freeze weights
            self.receptor_embedding.load_state_dict(
                {
                    'weight':
                        torch.nn.functional.one_hot(
                            torch.arange(
                                self.params['receptor_vocabulary_size']
                            )
                        )
                }
            )
            self.receptor_embedding.weight.requires_grad = False

        elif params.get('receptor_embedding', 'learned') == 'learned':
            self.receptor_embedding = nn.Embedding(
                self.params['receptor_vocabulary_size'],
                self.receptor_embedding_size,
                scale_grad_by_freq=params.get('embed_scale_grad', False),
            )
        else:
            assert params.get(
                'receptor_embedding', 'learned'
            ) == 'predefined', 'Choose either pretrained, one_hot, predefined \
             or learned as ligand_embedding. Defaults to learned'

        # Convolutions
        # TODO: Use nn.ModuleDict instead of the nn.Seq/OrderedDict
        self.ligand_convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'ligand_convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.ligand_filters, self.ligand_kernel_sizes)
                    )
                ]
            )
        )  # yapf: disable

        self.receptor_convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'receptor_convolutional_{index}',
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.receptor_filters, self.receptor_kernel_sizes)
                    )
                ]
            )
        )  # yapf: disable

        # Context attention
        self.context_attention_ligand_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'context_attention_ligand_{layer}',
                        ContextAttentionLayer(
                            self.ligand_hidden_sizes[layer],
                            self.params['ligand_padding_length'],
                            self.receptor_hidden_sizes[layer],
                            context_sequence_length=(
                                self.receptor_padding_length
                            ),
                            attention_size=self.ligand_attention_size,
                            individual_nonlinearity=params.get(
                                'context_nonlinearity', nn.Sequential()
                            ),
                            temperature=self.temperature,
                        ),
                    ) for layer in range(len(self.ligand_filters) + 1)
                ]
            )
        )

        self.context_attention_receptor_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f'context_attention_receptor_{layer}',
                        ContextAttentionLayer(
                            self.receptor_hidden_sizes[layer],
                            self.params['receptor_padding_length'],
                            self.ligand_hidden_sizes[layer],
                            context_sequence_length=self.ligand_padding_length,
                            attention_size=self.receptor_attention_size,
                            individual_nonlinearity=params.get(
                                'context_nonlinearity', nn.Sequential()
                            ),
                            temperature=self.temperature,
                        ),
                    ) for layer in range(len(self.receptor_filters) + 1)
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
                    ) for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = nn.Linear(self.hidden_sizes[-1], 1)
        if params.get('final_activation', True):
            self.final_dense = nn.Sequential(
                self.final_dense, ACTIVATION_FN_FACTORY['sigmoid']
            )

    def forward(self, ligand, receptors, confidence=False):
        """Forward pass through the biomodal MCA.

        Args:
            ligand (torch.Tensor): of type int and shape
                `[bs, ligand_padding_length]`.
            receptors (torch.Tensor): of type int and shape
                `[bs, receptor_padding_length]`.
            confidence (bool, optional) whether the confidence estimates are
                performed.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """
        # Embedding
        if self.ligand_embedding_type == 'predefined':
            embedded_ligand = ligand.to(torch.float)
        else:
            embedded_ligand = self.ligand_embedding(ligand.to(torch.int64))
        if self.receptor_embedding_type == 'predefined':
            embedded_receptor = receptors.to(torch.float)
        else:
            embedded_receptor = self.receptor_embedding(
                receptors.to(torch.int64)
            )

        # Convolutions
        encoded_ligand = [embedded_ligand] + [
            layer(torch.unsqueeze(embedded_ligand, 1)).permute(0, 2, 1)
            for layer in self.ligand_convolutional_layers
        ]
        encoded_receptor = [embedded_receptor] + [
            layer(torch.unsqueeze(embedded_receptor, 1)).permute(0, 2, 1)
            for layer in self.receptor_convolutional_layers
        ]

        # Context attention on ligand
        ligand_encodings, ligand_alphas = zip(
            *[
                layer(reference, context) for layer, reference, context in zip(
                    self.context_attention_ligand_layers,
                    encoded_ligand,
                    encoded_receptor,
                )
            ]
        )

        # Context attention on receptor
        receptor_encodings, receptor_alphas = zip(
            *[
                layer(reference, context) for layer, reference, context in zip(
                    self.context_attention_receptor_layers,
                    encoded_receptor,
                    encoded_ligand,
                )
            ]
        )

        # Concatenate all encodings
        encodings = torch.cat(
            [
                torch.cat(ligand_encodings, dim=1),
                torch.cat(receptor_encodings, dim=1),
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
            ligand_attention_weights = torch.mean(
                torch.cat(
                    [torch.unsqueeze(p, -1) for p in ligand_alphas], dim=-1
                ),
                dim=-1,
            )
            receptor_attention_weights = torch.mean(
                torch.cat(
                    [torch.unsqueeze(p, -1) for p in receptor_alphas], dim=-1
                ),
                dim=-1,
            )
            prediction_dict.update(
                {
                    'ligand_attention': ligand_attention_weights,
                    'receptor_attention': receptor_attention_weights,
                }
            )  # yapf: disable

            if confidence:
                augmenter = AugmentTensor(self.smiles_language)
                epistemic_conf = monte_carlo_dropout(
                    self,
                    regime='tensors',
                    tensors=(ligand, receptors),
                    repetitions=5,
                )
                aleatoric_conf = test_time_augmentation(
                    self,
                    regime='tensors',
                    tensors=(ligand, receptors),
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
