import os
import numpy as np
import pickle
import xgboost
from xgboost import XGBClassifier

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class RAScorerXGB:
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

        :param model_path: path to an XGBoost model (.json or legacy .pkl) file
        :type model_path: str
        """
        HERE = os.path.abspath(os.path.dirname(__file__))
        model_dir = os.path.join(HERE, "models/XGB_chembl_ecfp_counts")
        if model_path is None:
            json_model = os.path.join(model_dir, "model.json")
            model_path = json_model if os.path.exists(json_model) else os.path.join(model_dir, "model.pkl")
        model_path = os.fspath(model_path)

        if model_path.lower().endswith(".json"):
            self.xgb_model = XGBClassifier()
            self.xgb_model.load_model(model_path)
        elif model_path.lower().endswith(".pkl"):
            try:
                with open(model_path, "rb") as model_file:
                    self.xgb_model = pickle.load(model_file)
            except Exception as error:
                raise RuntimeError(
                    f"Unable to load XGBoost pickle '{model_path}' with XGBoost "
                    f"{xgboost.__version__}; use compatible package versions or a native .json model."
                ) from error
        else:
            raise ValueError("XGBoost model path must end in .json or .pkl")

    def ecfp(self, smiles):
        """
        Converts SMILES into a counted ECFP6 vector with features.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: ECFP6 counted vector with features
        :rtype: np.array
        """
        mol = Chem.MolFromSmiles(smiles)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        return generator.GetCountFingerprintAsNumPy(mol).astype(np.int32, copy=False)

    def predict(self, smiles):
        """
        Predicts score from SMILES.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: score
        :rtype: float
        """
        arr = self.ecfp(smiles)
        proba = self.xgb_model.predict_proba(arr.reshape(1, -1))
        return proba[0][1]

