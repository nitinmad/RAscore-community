## Installation 

Follow the steps in the defined order to avoid conflicts.

1. Use a Python 3.14 conda environment with XGBoost 3.4.x (for example):\
`conda create --name py314_test python=3.14`

or use an existing environment 

2. Install RDKit (if already installed skip this step)
```
conda install -c conda-forge rdkit -y
```

3. Clone the RAscore repository 
`git clone https://github.com/reymond-group/RAscore.git`

change directory to the repository
`cd RAscore`

4. Install RAscore
`python -m pip install -e .`

If you want to retrain models, or train your own models using the hyperparameter optimisation framework found in the 'model_building' folder, then the following should be installed in the environemnt also:\
`python -m pip install -e ".[retraining]"`

For PyTorch NN optimization and training, install `python -m pip install -e ".[retraining,nn]"`; this is the default backend. Both NN scripts also accept `--backend tensorflow`. For that backend, install `python -m pip install -e ".[retraining,nn-tensorflow]"`; it trains and writes a Keras `model.h5` without requiring PyTorch. If `.[nn]` is also installed, the builder additionally converts the trained weights to `model.pt`. NN inference requires `.[nn]` for PyTorch checkpoints, while loading legacy Keras models requires `.[nn-tensorflow]`.

For the XGBoost workflow, use the `XGBClassifier.json` config, which matches the shipped scorer's `ecfp_counts` descriptor. The optimizer selects parameters using cross-validation on the training data; the builder fits the final model and writes a native XGBoost `model.json` plus metrics. Keep the held-out test data out of optimization.

Extract `../data/data.zip` into `../data/` before running the config. The ChEMBL train and test CSV paths in the example config match the archive layout, and the candidate model is written under `../data/xgb_chembl_candidate/` rather than replacing the packaged model.

### Optional External Descriptors
The default XGBoost `ecfp_counts` workflow does not need these repositories. Clone and configure them only if using the corresponding SYBA, SCScore, or SA_Score descriptors. Run these commands from the repository root. First create the parent folder:

```powershell
New-Item -ItemType Directory -Force third_party | Out-Null
```

```sh
mkdir -p third_party
```

Then clone and install SYBA:

```sh
git clone https://github.com/lich-uct/syba.git third_party/syba
git clone https://github.com/connorcoley/scscore.git third_party/scscore
git clone --depth 1 --filter=blob:none --sparse https://github.com/rdkit/rdkit.git third_party/rdkit
git -C third_party/rdkit sparse-checkout set Contrib/SA_Score
python -m pip install -e third_party/syba
```

SCScore and SA_Score are imported from their cloned source directories. Add those directories to `PYTHONPATH` for the training session.

PowerShell:
```powershell
$env:PYTHONPATH = "$PWD/third_party/scscore/scscore;$PWD/third_party/rdkit/Contrib;$env:PYTHONPATH"
```

POSIX shell:
```sh
export PYTHONPATH="$PWD/third_party/scscore/scscore:$PWD/third_party/rdkit/Contrib${PYTHONPATH:+:$PYTHONPATH}"
```

## Training Models
### Configuration Files
The folder *example_classification_configs* contains example configuration files to train a variety of classifiers.\
The configuration file should be changed to reflect the paths of the training and test sets, and the output directory.\
A variety of descriptors have been implemented as found in the file *descriptors.py*. These can be specified in the configuration file.\

For instance consider the configuration file below:
* Specify: 
    * the path to the training data (.csv)
    * the path to the test data (.csv)
    * the path to the output directory
    * number of jobs for multiprocessing (for sklearn API)
    * number of cross vlaidation trails
    * training set and validation set size (test set) for internal cross validation
    * the descriptor:
        * ecfp_counts: counted extended connectivity fingerprint of radius 3
        * fcfp_counts: counted extended connectivity fingerprint of radius 3 with features
        * ecfp: extended connectivity fingerprint of radius 3 as a bit vector
        * maccs_keys: MACCS keys
        * features: molecular features as computed by RDkit i.e. molecular quantum numbers, number of rotatable bonds, molecualr weight, fraction of sp3 centres, number of heterocycles etc.
        * SA_score: Synthetic Accessibility score as published by Ertl et. al.
        * SC_score: Synthetic Complexity score as published by Coley et. al.
        * QED_score: Quantitative Estimate of Drug-Likeness as published by Bickerton et. al.
        * SYBA_score: Bayesian estimation of synthetic accessibility as published by Voršilák et. al.
        * custom descriptors can be added by implementing the descriptor in the *descriptors.py* file and adding a reference to both the optimiser and builder scripts
    * the algorithm:
        * AdaBoostClassifier
        * RandomForestClassifier
        * SVC
        * LogisticRegression
        * RidgeClassifier
        * BaggingClassifier
        * ModifiedNB
        * XGBClassifier
    * if running a feed forward neural network classifier specify and check out the example config file for optional parameters to explore:
        * DNNClassifier

```
{
"train_data": "<path-to-dataset>/train.csv",
"test_data": "<path-to-dataset>/test.csv",
"out_dir": "<path-to-output-directory>/RandomForestClassifier/",
"n_jobs": 6, 
"cross_validation": 5,
"train_size": 0.9,
"test_size": 0.1,
"descriptor": "fcfp_counts",
"n_trials": 500,
"algorithm": {"RandomForestClassifier": {
    "max_depth": {
        "low": 10,
        "high": 20
      },
      "n_estimators": {
        "low": 10,
        "high": 100
      },
      "max_features": ["sqrt", "log2"]
    }
}
}
```

### Optimisation
*Note: The hyperparameter optimisation does not use the test set specified but splits the training set into training and validation. The test set is then held out*\
To optimise and find favourable hyperparameters call:\
`python optimiser.py --config <path-to-config-file>`

or for the feed forward neural network call (PyTorch by default):\
`python optimiser_dnn.py --config <path-to-config-file> --backend torch`

To optimize using TensorFlow, pass `--backend tensorflow` and install the `nn-tensorflow` extra.

### Building 
Once optimisation is complete a .csv file containing the optimisation trials and a .json containing the best parameters and best values found will be present in the output folder specified in the config file.

*Note: Building the model uses the test set specified and has not been seen previously during optimisation*

To build the final model after optimization, call:\
`python builder.py --config <path-to-config-file> --best_params <path-to-best-parameter-file>`

or for the feed forward neural network call (PyTorch by default):\
`python builder_dnn.py --config <path-to-config-file> --best_params <path-to-best-parameter-file> --backend torch`

Use `--backend tensorflow` to train and save a Keras model instead.

To migrate the bundled legacy TensorFlow model into paired PyTorch and Keras artifacts, run `python model_building/convert_dnn.py` from the repository root in an environment where the existing SavedModel can be loaded.

