import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, QED

def _morgan_generator(radius, fp_size, use_features=False):
    if use_features:
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=fp_size,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

def ecfp(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    arr = _morgan_generator(radius, nBits).GetFingerprintAsNumPy(mol)
    return arr.astype(np.int32, copy=False)

def ecfp_counts(smiles, radius=3, useFeatures=False, useCounts=True):
    mol = Chem.MolFromSmiles(smiles)
    size = 2048
    generator = _morgan_generator(radius, size, use_features=useFeatures)
    if useCounts:
        arr = generator.GetCountFingerprintAsNumPy(mol)
    else:
        arr = generator.GetFingerprintAsNumPy(mol)
    return arr.astype(np.int32, copy=False)

def ecfp_features(smiles, radius=3, useFeatures=True, useCounts=False):
    return ecfp_counts(smiles, radius=radius, useFeatures=useFeatures, useCounts=useCounts)

def MACCS_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int32)
    arr[list(fp.GetOnBits())] = 1
    return arr

def feature_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.MQNs_(mol)
    
    fp.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    fp.append(rdMolDescriptors.CalcExactMolWt(mol))
    fp.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    fp.append(rdMolDescriptors.CalcFractionCSP3(mol))
    fp.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAliphaticRings((mol)))
    fp.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    fp.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    fp.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    fp.append(rdMolDescriptors.CalcNumRings(mol))
    fp.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    fp.append(rdMolDescriptors.CalcNumHeterocycles(mol))
    fp.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    fp.append(rdMolDescriptors.CalcTPSA(mol))
    
    return np.array(fp)

def sa_score(smi):
    try:
        from SA_Score import sascorer

        mol = Chem.MolFromSmiles(smi)
        return sascorer.calculateScore(mol)
    except:
        return np.nan

def QEDScore(smi):
     try:
        mol = Chem.MolFromSmiles(smi)
        return QED.qed(mol)
     except:
        return np.nan

def sc_score(smi, sc_model):
    try:
        (smiles, sc_score) = sc_model.get_score_from_smi(smi)
        return sc_score
    except:
        return np.nan