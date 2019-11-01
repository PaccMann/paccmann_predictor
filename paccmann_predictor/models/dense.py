import torch
import torch.nn as nn
from ..utils.layers import dense_layer
from ..utils.hyperparams import ACTIVATION_FN_FACTORY, LOSS_FN_FACTORY
from ..utils.utils import get_device


class Dense(nn.Module):
    """ This is a Dense model for validation """

    def __init__(self, params, *args, **kwargs):
        """Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the
                dense Decoder.
                TODO params should become actual arguments (use **params).

        Items in params:
            dense_sizes (list[int]): Number of neurons in the hidden layers.
            number_of_genes (int, optional): Number of -omics features of cell.
                Defaults to 2128.
            num_drug_features (int, optional): Number of features for molecule.
                Defaults to 512.
            activation_fn (string, optional): Activation function used in all
                layers for specification in ACTIVATION_FN_FACTORY.
                Defaults to 'relu'.
            batch_norm (bool, optional): Whether batch normalization is
                applied. Defaults to True.
            dropout (float, optional): Dropout probability in all
                except parametric layer. Defaults to 0.0.
            *args, **kwargs: positional and keyword arguments are ignored.

        Example params:
        ```
        {
            "dense_sizes": [2048, 1024, 512, 1],
            "dropout" : 0.1,
            "activation_fn": 'relu',
        }
        ```
        """

        super(Dense, self).__init__(*args, **kwargs)

        # Model Parameter
        self.device = get_device()
        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get('loss_fn', 'mse')]
        self.number_of_genes = params.get('number_of_genes', 2128)
        self.num_drug_features = params.get('num_drug_features', 512)
        self.hidden_sizes = params.get(
            'stacked_dense_hidden_sizes', [
                self.number_of_genes + self.num_drug_features, 1024, 512, 256,
                64
            ]
        )

        self.dropout = params.get('dropout', 0.0)
        self.act_fn = ACTIVATION_FN_FACTORY[
            params.get('activation_fn', 'relu')]

        self.dense_layers = [
            dense_layer(
                self.hidden_sizes[ind], self.hidden_sizes[ind + 1],
                self.act_fn, self.dropout
            ).to(self.device) for ind in range(len(self.hidden_sizes) - 1)
        ]

        self.final_dense = nn.Linear(self.hidden_sizes[-1], 1)

    def forward(self, fps, gep):
        """Forward pass through the dense model.

        Args:
            fps (torch.Tensor) of type int and shape `[bs, 512 (bits).
            gep (torch.Tensor): of shape `[bs, num_genes]`.

        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict

            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """

        inputs = torch.cat([fps.float(), gep], dim=1)

        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)
        prediction_dict = {'IC50': predictions}
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
