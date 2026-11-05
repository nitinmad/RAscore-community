"""TensorFlow model factory for the optional Keras NN training backend."""

from typing import Any


def build_keras_model(input_size: int, parameters: dict[str, Any]):
    """Build and compile a Keras model matching the PyTorch NN architecture."""
    try:
        from tensorflow import keras
    except ImportError as error:
        raise ImportError(
            "TensorFlow is required for the TensorFlow NN backend; "
            "install RAscore[nn-tensorflow]."
        ) from error

    inputs = keras.Input(shape=(input_size,))
    values = inputs
    for layer_index in range(1, int(parameters["num_layers"]) + 1):
        if layer_index == 1:
            units = int(parameters["layer_1"])
        else:
            units = int(parameters[f"units_{layer_index}"])
        values = keras.layers.Dense(
            units,
            activation=parameters[f"activation_{layer_index}"],
        )(values)
        dropout_rate = float(parameters.get(f"dropout_{layer_index}", 0.0))
        if dropout_rate:
            values = keras.layers.Dropout(dropout_rate)(values)

    outputs = keras.layers.Dense(1, activation="sigmoid", name="target")(values)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(
            learning_rate=float(parameters["learning_rate"])
        ),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model