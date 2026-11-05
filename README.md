# Retrosynthetic Accessibility (RA) score
 * RAscore is a score learned from the predictions of a computer aided synthesis planning tool (AiZynthfinder: https://github.com/MolecularAI/aizynthfinder). 
 * **RAscore is intended to be a binary score, indicating whether the underlying computer aided synthesis planning tool can find a route (1) or not (0) to a given compound.** 
 * The tool has been trained on 200,000 compounds from ChEMBL and so is limited to compounds within similar regions of chemical space. It is intended to predict the retrosynthetic accessibility of bioactive molecules. (Data can be found in `data/data.zip`)
 * Attempts to use the score on more exotic compounds such as those found in the GDB databases will not work: 
    * In this case the model will need to be switched to `GDBscore`, the corresponding models can be found in the models folder or downloaded from the pre-print server.

![alt text](RAscore/images/TOC.png)

## Known Issues
* The models may not generalise well in regions of chemical space that deviate from the training domains (ChEMBL and GDB subsets in this case), this is a problem inherent to ML based classifiers.
* The models have not been tested for the use case of molecular generation.
* **We highly recommend retraining the models on a sample of compounds representing your particular use case, the included data may be used to augment the data you generate**

## Installation 

Follow the steps in the defined order to avoid conflicts.

1. Create an environment for the current XGBoost runtime (Python 3.10 or newer):
```
conda create --name py314_test python=3.14
conda activate py314_test
```

or use an existing compatible environment.

2. Install RDKit (if already installed skip this step)
```
conda install -c conda-forge rdkit -y
```

3. Install RAscore 
```
pip install git+https://github.com/reymond-group/RAscore.git@master
```
The models.zip file will have to be downloaded and after unzipping the path to the relevant model will need to be specified on model instantiation.

or 

Clone and install the repository using (models should be included):
```
git clone https://github.com/reymond-group/RAscore.git
pip install --editable .
```
If the models are not automatically included check that the models.zip file exists and unzip it into the desired location.

### Model Compatibility
The bundled XGBoost `model.pkl` was created with XGBoost 1.0.2 and scikit-learn 0.22.1. Until it is replaced by a retrained native `model.json`, loading that legacy artifact requires its original compatible environment. Retrain using XGBoost 3.4.x before relying on the Python 3.14 runtime.

The neural-network scorer uses PyTorch checkpoints (`.pt`) by default. Install the optional runtime with `python -m pip install -e ".[nn]"`. Keras `.h5` models and paired Keras exports can be loaded or created with `python -m pip install -e ".[nn-tensorflow]"`; TensorFlow is not needed for PyTorch inference. This checkout currently contains the legacy TensorFlow SavedModel, so generate its paired `model.pt` and `model.h5` artifacts before using the bundled NN by running `python model_building/convert_dnn.py` in an environment that can load that SavedModel.

## Usage
### Importing in Python
The NN scorer uses PyTorch; the XGBoost scorer uses its native model format.

To walk through the example in a jupyter notebook refer to `rascore_usage.ipynb`

```
from RAscore import RAscore_NN #For PyTorch NN models
from RAscore import RAscore_XGB #For XGB based models

nn_scorer = RAscore_NN.RAScorerNN() 
xgb_scorer = RAscore_XGB.RAScorerXGB()

#Imatinib mesylate
imatinib_mesylate = 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5.CS(=O)(=O)O'
nn_scorer.predict(imatinib_mesylate)
0.99522984

xgb_scorer.predict(imatinib_mesylate)
0.99259007

#Omeprazole
omeprazole = 'CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC'
nn_scorer.predict(omeprazole)
0.99999106

xgb_scorer.predict(omeprazole)
0.9556329

#Morphine - Illustrates problem synthesis planning tools face with complex ring systems
morphine = 'CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5'
nn_scorer.predict(morphine)
8.316945e-07

xgb_scorer.predict(morphine)
0.0028359715
```

### Command Line Interface
A command line interface is provided which allows batch processing and enables the flexibility of specifying models.\
```
Usage: RAscore [OPTIONS]

Example: A set of smiles `test.smi` are provided

`RAscore -f test.smi -c SMILES -o test.csv`

Default Model: XGBoost using ChEMBL and ECFP4 counts with features

Options:
  -f, --file_path TEXT      Absolute path to input file containing one SMILES
                            on each line. The column should be labelled
                            "SMILES" or if another header is used, specify it
                            as an option

  -c, --column_header TEXT  The name given to the singular column in the file
                            which contains the SMILES. The column must be
                            named.

  -o, --output_path TEXT    Output file path
    -m, --model_path TEXT     Model path: .pt for PyTorch NN, .h5 for Keras NN,
                                                        or .json/.pkl for XGBoost

  --help                    Show this message and exit.
```
Further RAscore models are contained in the `RAscore/models/models.zip` folder if you wish to specify a different model than the default:
* RAscore
    * DNN_chembl_fcfp_counts
    * XGB_chembl_ecfp_counts
* GDBscore
    * DNN_gdbchembl_fcfp_counts
    * XGB_gdbchembl_ecfp_counts

## Retraining  
To retrain the XGBoost model or use the hyperparameter optimisation framework, install the retraining dependencies as well:
`python -m pip install -e ".[retraining]"`

For PyTorch NN training, install `python -m pip install -e ".[retraining,nn]"`. Both NN scripts also support `--backend tensorflow`; that path requires `python -m pip install -e ".[retraining,nn-tensorflow]"` and can train without PyTorch. The TensorFlow builder writes `.h5`, and also writes `.pt` when the `nn` extra is installed.

The default XGBoost `ecfp_counts` retraining workflow does not need external descriptor repositories. For optional SYBA, SCScore, or SA_Score descriptors, see the clone and setup instructions in [`model_building/README.md`](model_building/README.md#optional-external-descriptors).

Please refer to the `model_building` folder for further information about retraining.

## Performance on Test Set
* Test set contains ca. 20,000 compounds from ChEMBL
* The model was able to separate clusters of solved/unsolved compounds as found by computing the average linkage
* RAscore can better differentiate between solved/unsolved compounds than existing methods.
![alt text](RAscore/images/RA_Score_histogram.png)

## Computation of Average Linkage 
![alt text](RAscore/images/average_linkage.png)

## Citation
The models have been published in Chemical Science

Thakkar, A.; Chadimová, V.; Bjerrum, E. J.; Engkvist, O.; Reymond, J.-L. Retrosynthetic Accessibility Score (RAscore) – Rapid Machine Learned Synthesizability Classification from AI Driven Retrosynthetic Planning. Chem. Sci. 2021. https://doi.org/10.1039/d0sc05401a

