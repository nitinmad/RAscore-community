"""Test the default models can be loaded."""

import math
import os
import tempfile
import unittest


class TestLoad(unittest.TestCase):
    """Test the default models can be loaded."""

    def test_import_rascore(self):
        """Test import is possible."""
        import RAscore
        print(RAscore)

    def test_import_rdkit(self):
        """Test importing RDKit works."""

    def test_nn(self):
        """Test loading the PyTorch NN and checkpoint round-trip."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the XGBoost runtime environment")

        from RAscore import RAscore_NN
        from RAscore.nn_torch import save_checkpoint

        nn_scorer = RAscore_NN.RAScorerNN()
        smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O"
        expected = nn_scorer.predict(smiles)
        self.assertTrue(math.isfinite(expected))
        self.assertGreaterEqual(expected, 0.0)
        self.assertLessEqual(expected, 1.0)

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, "round_trip.pt")
            save_checkpoint(nn_scorer.nn_model, model_path)
            reloaded = RAscore_NN.RAScorerNN(model_path)

        self.assertAlmostEqual(expected, reloaded.predict(smiles), places=6)

    def test_nn_tensorflow_companion(self):
        """Test that the optional Keras companion matches the PyTorch model."""
        try:
            import tensorflow
        except ImportError:
            self.skipTest("TensorFlow is not installed")

        from RAscore import RAscore_NN

        model_dir = os.path.join(
            os.path.dirname(RAscore_NN.__file__),
            "models",
            "DNN_chembl_fcfp_counts",
        )
        tensorflow_model = os.path.join(model_dir, "model.h5")
        if not os.path.exists(tensorflow_model):
            self.skipTest("The TensorFlow companion has not been generated")

        smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O"
        torch_score = RAscore_NN.RAScorerNN().predict(smiles)
        tensorflow_score = RAscore_NN.RAScorerNN(tensorflow_model).predict(smiles)
        self.assertAlmostEqual(torch_score, tensorflow_score, places=5)

    def test_tensorflow_training_model_factory(self):
        """Test the optional TensorFlow training backend model factory."""
        try:
            import tensorflow
        except ImportError:
            self.skipTest("TensorFlow is not installed")

        from RAscore.nn_keras import build_keras_model
        import numpy as np

        parameters = {
            "layer_1": 4,
            "activation_1": "linear",
            "num_layers": 2,
            "units_2": 3,
            "activation_2": "relu",
            "dropout_2": 0.1,
            "learning_rate": 0.001,
        }
        model = build_keras_model(5, parameters)
        output = model(np.zeros((2, 5), dtype=np.float32), training=False)
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_xgb(self):
        """Test loading the XGB models."""
        from RAscore import RAscore_XGB

        xgb_scorer = RAscore_XGB.RAScorerXGB()

    def test_xgb_predict_and_json_round_trip(self):
        """Test prediction and native model serialization."""
        from RAscore import RAscore_XGB

        smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O"
        scorer = RAscore_XGB.RAScorerXGB()
        expected = scorer.predict(smiles)
        self.assertTrue(math.isfinite(expected))
        self.assertGreaterEqual(expected, 0.0)
        self.assertLessEqual(expected, 1.0)

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = os.path.join(model_dir, "round_trip.json")
            scorer.xgb_model.save_model(model_path)
            reloaded = RAscore_XGB.RAScorerXGB(model_path)

        self.assertAlmostEqual(expected, reloaded.predict(smiles), places=6)
