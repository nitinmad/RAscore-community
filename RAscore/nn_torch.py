"""PyTorch model and framework-neutral serialization helpers for the NN scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from sklearn.metrics import roc_auc_score


_ACTIVATIONS = {
    "elu": nn.ELU,
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}


class RAScoreNN(nn.Module):
    """Dense binary classifier matching the repository's Keras NN layout."""

    def __init__(self, input_size: int, parameters: dict[str, Any]):
        super().__init__()
        self.input_size = input_size
        self.model_config = dict(parameters)
        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        previous_size = input_size
        for layer_index in range(1, int(parameters["num_layers"]) + 1):
            if layer_index == 1:
                units = int(parameters["layer_1"])
            else:
                units = int(parameters[f"units_{layer_index}"])

            activation_name = parameters[f"activation_{layer_index}"].lower()
            try:
                activation = _ACTIVATIONS[activation_name]
            except KeyError as error:
                raise ValueError(f"Unsupported NN activation: {activation_name}") from error

            dropout_rate = float(parameters.get(f"dropout_{layer_index}", 0.0))
            self.hidden_layers.append(nn.Linear(previous_size, units))
            self.activations.append(activation())
            self.dropouts.append(nn.Dropout(dropout_rate))
            previous_size = units

        self.output_layer = nn.Linear(previous_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs
        for linear, activation, dropout in zip(
            self.hidden_layers, self.activations, self.dropouts
        ):
            values = dropout(activation(linear(values)))
        return torch.sigmoid(self.output_layer(values))


def save_checkpoint(model: RAScoreNN, path: str | Path) -> None:
    """Save architecture metadata and a PyTorch state dict."""
    torch.save(
        {
            "format_version": 1,
            "input_size": model.input_size,
            "parameters": model.model_config,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path) -> RAScoreNN:
    """Load a PyTorch NN checkpoint on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported PyTorch NN checkpoint format")
    model = RAScoreNN(checkpoint["input_size"], checkpoint["parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def train_model(
    model: RAScoreNN,
    features: Any,
    labels: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float = 0.0,
) -> RAScoreNN:
    """Train a model with shuffled mini-batches and binary cross entropy."""
    inputs = torch.as_tensor(features, dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1)
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")
    split_index = int(len(inputs) * (1.0 - validation_split))
    if split_index == 0:
        raise ValueError("validation_split leaves no training examples")
    inputs = inputs[:split_index]
    targets = targets[:split_index]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        for batch_indices in torch.randperm(len(inputs)).split(batch_size):
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            optimizer.zero_grad()
            probabilities = model(batch_inputs)
            loss = nn.functional.binary_cross_entropy(probabilities, batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate_model(model: RAScoreNN, features: Any, labels: Any) -> dict[str, float]:
    """Return binary cross entropy and ROC AUC for a dataset."""
    inputs = torch.as_tensor(features, dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        probabilities = model(inputs)
        loss = nn.functional.binary_cross_entropy(probabilities, targets).item()
    predicted = probabilities.cpu().numpy().reshape(-1)
    actual = targets.cpu().numpy().reshape(-1)
    return {"loss": loss, "auc": float(roc_auc_score(actual, predicted))}


def convert_keras_model(
    keras_model: Any,
    input_size: int,
    parameters: dict[str, Any],
    path: str | Path,
) -> RAScoreNN:
    """Copy weights from a loaded Keras dense network into a PyTorch checkpoint."""
    model = RAScoreNN(input_size, parameters)
    torch_layers = [*model.hidden_layers, model.output_layer]
    keras_layers = [
        layer for layer in keras_model.layers if len(layer.get_weights()) == 2
    ]
    if len(keras_layers) != len(torch_layers):
        raise ValueError("Keras model layer count does not match NN architecture")

    with torch.no_grad():
        for keras_layer, torch_layer in zip(keras_layers, torch_layers):
            kernel, bias = keras_layer.get_weights()
            if tuple(kernel.T.shape) != tuple(torch_layer.weight.shape):
                raise ValueError("Keras model layer shapes do not match NN architecture")
            torch_layer.weight.copy_(torch.from_numpy(kernel.T.copy()))
            torch_layer.bias.copy_(torch.from_numpy(bias.copy()))

    model.eval()
    save_checkpoint(model, path)
    return model


def export_keras_model(model: RAScoreNN, path: str | Path) -> None:
    """Export the trained weights to a Keras HDF5 model."""
    try:
        from tensorflow import keras
    except ImportError as error:
        raise ImportError(
            "TensorFlow is required to export the Keras companion model; "
            "install RAscore[nn-tensorflow]."
        ) from error

    parameters = model.model_config
    inputs = keras.Input(shape=(model.input_size,))
    values = inputs
    with torch.no_grad():
        for layer_index, (linear, dropout) in enumerate(
            zip(model.hidden_layers, model.dropouts), start=1
        ):
            activation = parameters[f"activation_{layer_index}"].lower()
            dense = keras.layers.Dense(linear.out_features, activation=activation)
            values = dense(values)
            dense.set_weights(
                [
                    linear.weight.detach().cpu().numpy().T,
                    linear.bias.detach().cpu().numpy(),
                ]
            )
            if dropout.p:
                values = keras.layers.Dropout(dropout.p)(values)

        output_layer = keras.layers.Dense(1, activation="sigmoid")
        output = output_layer(values)
        output_layer.set_weights(
            [
                model.output_layer.weight.detach().cpu().numpy().T,
                model.output_layer.bias.detach().cpu().numpy(),
            ]
        )

    keras_model = keras.Model(inputs=inputs, outputs=output)
    keras_model.save(path, include_optimizer=False)
