import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class RAScorerNN:
    """
    Prediction of machine learned retrosynthetic accessibility score
    The RAScore is calculated based on the predictions made on 200,000 compounds sampled from ChEMBL.
    The compounds were subjected to retrosynthetic analysis using a CASP tool (AiZynthfinder) and output used as labels to train a binary classifier.

    If the compounds are ChEMBL like use the RAscore models.
    Else if the compounds are more exotic, are small fragments, or closely resemble GDB. GBDscore may give a better result.

    This class facilitates predictions from the resulting model.
    """

    def __init__(self, model_path=None):
        """
        Loads the model.

        :param model_path: path to a PyTorch (.pt) or Keras (.h5) model
        """
        HERE = os.path.abspath(os.path.dirname(__file__))
        default_model = os.path.join(
            HERE, "models/DNN_chembl_fcfp_counts/model.pt"
        )
        model_path = model_path or default_model
        extension = os.path.splitext(model_path)[1].lower()
        if extension == ".pt":
            try:
                from RAscore.nn_torch import load_checkpoint
            except ImportError as error:
                raise ImportError(
                    "PyTorch is required to load NN checkpoints; install RAscore[nn]."
                ) from error

            self.nn_model = load_checkpoint(model_path)
            self._backend = "torch"
        elif extension in (".h5", ".hdf5") or os.path.isdir(model_path):
            try:
                from tensorflow import keras
            except ImportError as error:
                raise ImportError(
                    "TensorFlow is required to load Keras NN models; "
                    "install RAscore[nn-tensorflow]."
                ) from error
            self.nn_model = keras.models.load_model(model_path)
            self._backend = "tensorflow"
        else:
            raise ValueError("Neural-network model must be a .pt, .h5, or .hdf5 file")

    def ecfp_counts(self, smiles):
        """
        Converts SMILES into a counted ECFP6 vector with features.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: ECFP6 counted vector with features
        :rtype: np.array
        """
        mol = Chem.MolFromSmiles(smiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=3,
            fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        return generator.GetCountFingerprintAsNumPy(mol).astype(np.int32, copy=False)

    def predict(self, smiles):
        """
        Predicts score from SMILES.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: score
        :rtype: float
        """
        try:
            arr = self.ecfp_counts(smiles)
        except ValueError:
            print("SMILES could not be converted to ECFP6 count vector")
            return float("NaN")

        try:
            if self._backend == "torch":
                import torch

                self.nn_model.eval()
                with torch.no_grad():
                    values = torch.from_numpy(arr.reshape(1, -1).astype(np.float32))
                    return float(self.nn_model(values).item())
            proba = self.nn_model.predict(arr.reshape(1, -1), verbose=0)
            return float(proba[0][0])
        except:
            print("Prediction not possible")
            return float("NaN")

