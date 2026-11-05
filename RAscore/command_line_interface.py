import click
import logging
import os
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--file_path",
    "-f",
    help='Absolute path to input file containing one SMILES on each line. The column should be labelled "SMILES" or if another header is used, specify it as an option',
)
@click.option(
    "--column_header",
    "-c",
    default="SMILES",
    help="The name given to the singular column in the file which contains the SMILES. The column must be named.",
)
@click.option("--output_path", "-o", help="Output file path")
@click.option(
    "--model_path",
    "-m",
    help="Model path: .pt for PyTorch NN, .h5 for Keras NN, or .json/.pkl for XGBoost",
)
def main(file_path: str, output_path: str, model_path: str, column_header: str) -> None:
    if file_path.endswith(".txt"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".smi"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep="\t")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        ValueError("Unrecognized file type")

    if not model_path or model_path.lower().endswith((".pkl", ".json")):
        from RAscore import RAscore_XGB  # For XGB based models

        scorer = RAscore_XGB.RAScorerXGB(model_path)

    elif model_path.lower().endswith((".pt", ".h5", ".hdf5")):
        from RAscore import RAscore_NN  # For PyTorch and Keras NN models

        scorer = RAscore_NN.RAScorerNN(model_path)
    else:
        raise click.BadParameter(
            "model must end in .pt, .h5, or .hdf5 for NN models, "
            "or .json/.pkl for XGBoost",
            param_hint="--model_path",
        )

    tqdm.pandas()
    df["RAscore"] = df[column_header].progress_apply(scorer.predict)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
