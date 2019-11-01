import torch
import torch.nn as nn


def get_device():
    return torch.device('cuda' if cuda() else 'cpu')


def cuda():
    return torch.cuda.is_available()


def to_np(x):
    return x.data.cpu().numpy()


def attention_list_to_matrix(coding_tuple, dim=2):
    """[summary]

    Args:
        coding_tuple (list((torch.Tensor, torch.Tensor))): iterable of
            (outputs, att_weights) tuples coming from the attention function
        dim (int, optional): The dimension along which expansion takes place to
            concatenate the attention weights. Defaults to 2.

    Returns:
        (torch.Tensor, torch.Tensor): raw_coeff, coeff

        raw_coeff: with the attention weights of all multiheads and
            convolutional kernel sizes concatenated along the given dimension,
            by default the last dimension.
        coeff: where the dimension is collapsed by averaging.
    """
    raw_coeff = torch.cat(
        [torch.unsqueeze(tpl[1], 2) for tpl in coding_tuple], dim=dim
    )
    return raw_coeff, torch.mean(raw_coeff, dim=dim)


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""
    def forward(self, data):
        return torch.squeeze(data)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""
    def __init__(self, axis):
        super(Unsqueeze, self).__init__()
        self.axis = axis

    def forward(self, data):
        return torch.unsqueeze(data, self.axis)
