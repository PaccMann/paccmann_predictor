"""Custom layers implementation."""
from collections import OrderedDict

import torch
import torch.nn as nn

from .utils import Squeeze, get_device, Temperature, Unsqueeze

DEVICE = get_device()


def dense_layer(
    input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.0
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(input_size, hidden_size)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(hidden_size)
                    if batch_norm else nn.Identity(),
                ),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
            ]
        )
    )


def dense_attention_layer(
    number_of_features: int,
    temperature: float = 1.0,
    dropout=0.0
) -> nn.Sequential:
    """Attention mechanism layer for dense inputs.

    Args:
        number_of_features (int): Size to allocate weight matrix.
        temperature (float): Softmax temperature parameter (0, inf). Lower
            temperature (< 1) result in a more descriminative/spiky softmax,
            higher temperature (> 1) results in a smoother attention.
    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ('dense', nn.Linear(number_of_features, number_of_features)),
                ('dropout', nn.Dropout(p=dropout)),
                ('temperature', Temperature(temperature)),
                ('softmax', nn.Softmax(dim=-1)),
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.0,
    input_channels=1,
):
    """Convolutional layer.

    Args:
        num_kernel (int): Number of convolution kernels.
        kernel_size (tuple[int, int]): Size of the convolution kernels.
        act_fn (callable): Functional of the nonlinear activation.
        batch_norm (bool): whether batch normalization is applied.
        dropout (float): Probability for each input value to be 0.
        input_channels (int): Number of input channels (defaults to 1).

    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    'convolve',
                    torch.nn.Conv2d(
                        input_channels,  # channel_in
                        num_kernel,  # channel_out
                        kernel_size,  # kernel_size
                        padding=[kernel_size[0] // 2,
                                 0],  # pad for valid conv.
                    ),
                ),
                ('squeeze', Squeeze()),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(num_kernel)
                    if batch_norm else nn.Identity(),
                ),
            ]
        )
    )


class ContextAttentionLayer(nn.Module):
    """
    Implements context attention as in the PaccMann paper (Figure 2C) in
    Molecular Pharmaceutics.
    With the additional option of having a hidden size in the context.
    NOTE:
    In tensorflow, weights were initialized from N(0,0.1). Instead, pytorch
    uses U(-stddev, stddev) where stddev=1./math.sqrt(weight.size(1)).
    """

    def __init__(
        self,
        reference_hidden_size: int,
        reference_sequence_length: int,
        context_hidden_size: int,
        context_sequence_length: int = 1,
        attention_size: int = 16,
        individual_nonlinearity: type = nn.Sequential(),
        temperature: float = 1.0,
    ):
        """Constructor
        Arguments:
            reference_hidden_size (int): Hidden size of the reference input
                over which the attention will be computed (H).
            reference_sequence_length (int): Sequence length of the reference
                (T).
            context_hidden_size (int): This is either simply the amount of
                features used as context (G) or, if the context is a sequence
                itself, the hidden size of each time point.
            context_sequence_length (int): Hidden size in the context, useful
                if context is also textual data, i.e. coming from nn.Embedding.
                Defaults to 1.
            attention_size (int): Hyperparameter of the attention layer,
                defaults to 16.
            individual_nonlinearities (type): This is an optional
                nonlinearity applied to each projection. Defaults to
                nn.Sequential(), i.e. no nonlinearity. Otherwise it expects a
                torch.nn activation function, e.g. nn.ReLU().
            temperature (float): Temperature parameter to smooth or sharpen the
                softmax. Defaults to 1. Temperature > 1 flattens the
                distribution, temperature below 1 makes it spikier.
        """
        super().__init__()

        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity
        self.temperature = temperature

        # Project the reference into the attention space
        self.reference_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(reference_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Project the context into the attention space
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        'projection',
                        nn.Linear(context_hidden_size, attention_size),
                    ),
                    ('act_fn', individual_nonlinearity),
                ]
            )
        )  # yapf: disable

        # Optionally reduce the hidden size in context
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'projection',
                            nn.Linear(
                                context_sequence_length,
                                reference_sequence_length,
                            ),
                        ),
                        ('act_fn', individual_nonlinearity),
                    ]
                )
            )  # yapf: disable
        else:
            self.context_hidden_projection = nn.Sequential()

        self.alpha_projection = nn.Sequential(
            OrderedDict(
                [
                    ('projection', nn.Linear(attention_size, 1, bias=False)),
                    ('squeeze', Squeeze()),
                    ('temperature', Temperature(self.temperature)),
                    ('softmax', nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(
        self,
        reference: torch.Tensor,
        context: torch.Tensor,
        average_seq: bool = True
    ):
        """
        Forward pass through a context attention layer
        Arguments:
            reference (torch.Tensor): This is the reference input on which
                attention is computed. Shape: bs x ref_seq_length x ref_hidden_size
            context (torch.Tensor): This is the context used for attention.
                Shape: bs x context_seq_length x context_hidden_size
            average_seq (bool): Whether the filtered attention is averaged over the
                sequence length.
                NOTE: This is recommended to be True, however if the ref_hidden_size
                is 1, this can be used to prevent collapsing to a single float.
                Defaults to True.
        Returns:
            (output, attention_weights):  A tuple of two Tensors, first one
                containing the reference filtered by attention (shape:
                bs x ref_hidden_size) and the second one the
                attention weights (bs x ref_seq_length).
                NOTE: If average_seq is False, the output is: bs x ref_seq_length
        """
        assert len(reference.shape) == 3, 'Reference tensor needs to be 3D'
        assert len(context.shape) == 3, 'Context tensor needs to be 3D'

        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(
            self.context_projection(context).permute(0, 2, 1)
        ).permute(0, 2, 1)
        alphas = self.alpha_projection(
            torch.tanh(reference_attention + context_attention)
        )

        output = reference * torch.unsqueeze(alphas, -1)
        output = torch.sum(output, 1) if average_seq else torch.squeeze(output)

        return output, alphas


def gene_projection(num_genes, attention_size, ind_nonlin=nn.Sequential()):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(num_genes, attention_size)),
                ('act_fn', ind_nonlin),
                ('expand', Unsqueeze(1)),
            ]
        )
    ).to(DEVICE)


def smiles_projection(
    smiles_hidden_size, attention_size, ind_nonlin=nn.Sequential()
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(smiles_hidden_size, attention_size)),
                ('act_fn', ind_nonlin),
            ]
        )
    ).to(DEVICE)


def alpha_projection(attention_size):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(attention_size, 1, bias=False)),
                ('squeeze', Squeeze()),
                ('softmax', nn.Softmax(dim=1)),
            ]
        )
    ).to(DEVICE)
