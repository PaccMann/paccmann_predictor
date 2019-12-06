"""Custom layers implementation.

Inspired by Bahdanau attention, the following implements a contextual attention
mechanism. The attention weights specify how well each token of the encoded
SMILES (e.g. bRNN, raw embedding, conv_output) targets the genes.

NOTE:
gene_projection and smiles_projection are used to project genes and SMILES into
    common space. Then in forward() these two are added and given through a
    tanh before the alpha_projection is applied to get the attention weights.
NOTE:
In tensorflow, weights were initialized from N(0,0.1). Instead, pytorch uses
    U(-stddev, stddev) where stddev=1./math.sqrt(weight.size(1)).
"""
from collections import OrderedDict

import torch
import torch.nn as nn

from .utils import Squeeze, Unsqueeze, get_device

DEVICE = get_device()


def dense_layer(
    input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(input_size, hidden_size)),
                (
                    'batch_norm', nn.BatchNorm1d(hidden_size)
                    if batch_norm else nn.Identity()
                ),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
            ]
        )
    )


def dense_attention_layer(number_of_features):
    """Attention mechanism layer for dense inputs.

    Args:
        number_of_features (int): Size to allocate weight matrix.
    Returns:
        callable: a function that can be called with inputs.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ('dense', nn.Linear(number_of_features, number_of_features)),
                ('softmax', nn.Softmax())
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=nn.ReLU(),
    batch_norm=False,
    dropout=0.,
    input_channels=1
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
                        padding=[kernel_size[0] // 2, 0]  # pad for valid conv.
                    )
                ),
                ('squeeze', Squeeze()),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
                (
                    'batch_norm',
                    nn.BatchNorm1d(num_kernel) if batch_norm else nn.Identity()
                )
            ]
        )
    )


def gene_projection(num_genes, attention_size, ind_nonlin=nn.Sequential()):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(num_genes, attention_size)),
                ('act_fn', ind_nonlin), ('expand', Unsqueeze(1))
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
                ('act_fn', ind_nonlin)
            ]
        )
    ).to(DEVICE)


def alpha_projection(attention_size):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(attention_size, 1, bias=False)),
                ('squeeze', Squeeze()), ('softmax', nn.Softmax(dim=1))
            ]
        )
    ).to(DEVICE)
