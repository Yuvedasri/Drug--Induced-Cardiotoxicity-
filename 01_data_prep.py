"""
Step 1-3 of workflow:
  - Select SMILES + Assay outcome
  - Remove missing values
  - Remove inconclusive samples (n/a for Karim, already binary)
Primary dataset: hERG_Karim_Dataset.xlsx (clean, balanced, binary)
"""
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # silence RDKit parse warnings, we handle failures explicitly

DATA_DIR = "data"
OUT_DIR = "outputs"

def load_karim():
    df = pd.read_excel(f"{DATA_DIR}/hERG_Karim_Dataset.xlsx")
    df = df.rename(columns={"Drug": "SMILES", "Y": "label"})
    df = df[["Drug_ID", "SMILES", "label"]].dropna(subset=["SMILES", "label"])
    df["label"] = df["label"].astype(int)
    return df

def validate_smiles(df):
    """Drop rows where RDKit cannot parse the SMILES into a valid molecule."""
    valid_mask = []
    canonical_smiles = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            valid_mask.append(False)
            canonical_smiles.append(None)
        else:
            valid_mask.append(True)
            canonical_smiles.append(Chem.MolToSmiles(mol))
    df = df.copy()
    df["valid"] = valid_mask
    df["canonical_smiles"] = canonical_smiles
    n_before = len(df)
    df = df[df["valid"]].drop(columns=["valid"])
    n_after = len(df)
    print(f"RDKit parsing: {n_before} -> {n_after} ({n_before - n_after} unparsable dropped)")
    return df

def dedupe(df):
    """Remove duplicate compounds by canonical SMILES; flag label conflicts."""
    n_before = len(df)
    conflict = df.groupby("canonical_smiles")["label"].nunique()
    conflicting_smiles = conflict[conflict > 1].index
    n_conflict = len(conflicting_smiles)
    if n_conflict:
        print(f"WARNING: {n_conflict} compounds have conflicting labels across duplicates -- dropping them")
        df = df[~df["canonical_smiles"].isin(conflicting_smiles)]
    df = df.drop_duplicates(subset="canonical_smiles").reset_index(drop=True)
    print(f"Dedup: {n_before} -> {len(df)}")
    return df

if __name__ == "__main__":
    df = load_karim()
    print(f"Loaded Karim dataset: {df.shape}")
    print(df["label"].value_counts())

    df = validate_smiles(df)
    df = dedupe(df)

    print("\nFinal class balance:")
    print(df["label"].value_counts())
    print(f"Final dataset size: {len(df)}")

    df.to_csv(f"{OUT_DIR}/clean_herg_dataset.csv", index=False)
    print(f"\nSaved cleaned dataset to {OUT_DIR}/clean_herg_dataset.csv")