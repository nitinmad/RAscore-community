"""Convert the bundled Keras DNN into paired PyTorch and Keras artifacts."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from RAscore.nn_torch import convert_keras_model, export_keras_model


def main():
    model_dir = os.path.join(
        PROJECT_ROOT, "RAscore", "models", "DNN_chembl_fcfp_counts"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", default=os.path.join(model_dir, "model.tf"))
    parser.add_argument(
        "--parameters", default=os.path.join(model_dir, "best_params.json")
    )
    parser.add_argument("--output-dir", default=model_dir)
    args = parser.parse_args()

    try:
        from tensorflow import keras
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is required to convert the bundled model; "
            "install RAscore[nn-tensorflow]."
        ) from error

    with open(args.parameters, encoding="utf-8") as parameter_file:
        parameters = json.load(parameter_file)

    keras_model = keras.models.load_model(args.keras_model)
    os.makedirs(args.output_dir, exist_ok=True)
    torch_model = convert_keras_model(
        keras_model,
        input_size=int(keras_model.input_shape[-1]),
        parameters=parameters,
        path=os.path.join(args.output_dir, "model.pt"),
    )
    export_keras_model(torch_model, os.path.join(args.output_dir, "model.h5"))


if __name__ == "__main__":
    main()