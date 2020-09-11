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


def get_log_molar(y, ic50_max=None, ic50_min=None):
    """
    Converts PaccMann predictions from [0,1] to log(micromolar) range.
    """
    return y * (ic50_max - ic50_min) + ic50_min


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return torch.unsqueeze(data, self.dim)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature
