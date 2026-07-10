"""
External validation: evaluate the trained Causal-GAT (base GAT weights) on the
independent Tox21 hERG aggregated dataset. This tests generalization to a
completely different data source/assay campaign than the Karim training set.

Important: we check for overlap with the training set by canonical SMILES and
exclude any compounds also present in Karim training data, to avoid leakage.
"""
import sys
sys.path.insert(0, "/home/claude/project/src")

import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model import LightweightGAT
from importlib import import_module
graph_mod = import_module("02_graph_construction")

DATA_DIR = "/home/claude/project/data"
OUT_DIR = "/home/claude/project/outputs"


def load_and_clean_tox21():
    df = pd.read_csv(f"{DATA_DIR}/tox21-herg-u2os-p1.aggregrated.txt", sep="\t")
    # NOTE: this file has a trailing-tab column offset -- the header labels don't
    # line up with the data. Verified mapping: true SMILES lives in the column
    # pandas labels SAMPLE_NAME, and the true assay outcome lives in the column
    # pandas labels SAMPLE_DATA_TYPE (confirmed against the known class counts).
    df = df[["SAMPLE_NAME", "SAMPLE_DATA_TYPE"]].dropna()
    df = df.rename(columns={"SAMPLE_NAME": "SMILES", "SAMPLE_DATA_TYPE": "ASSAY_OUTCOME"})

    label_map = {
        "inactive": 0,
        "active antagonist": 1,
        "active agonist": 1,
    }
    df = df[df["ASSAY_OUTCOME"].isin(label_map.keys())].copy()
    df["label"] = df["ASSAY_OUTCOME"].map(label_map)
    print(f"Tox21 after dropping inconclusive: {len(df)} rows")
    print(df["label"].value_counts())

    # canonicalize + validate
    canon, valid = [], []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            canon.append(None)
            valid.append(False)
        else:
            canon.append(Chem.MolToSmiles(mol))
            valid.append(True)
    df["canonical_smiles"] = canon
    df = df[valid]

    # drop internal duplicates / conflicts
    conflict = df.groupby("canonical_smiles")["label"].nunique()
    df = df[~df["canonical_smiles"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates(subset="canonical_smiles").reset_index(drop=True)
    print(f"After validation/dedup: {len(df)} rows")
    return df


def main():
    device = torch.device("cpu")

    # Load training set canonical SMILES to check leakage
    train_df = pd.read_csv(f"{OUT_DIR}/clean_herg_dataset.csv")
    train_smiles_set = set(train_df["canonical_smiles"])

    tox21_df = load_and_clean_tox21()

    overlap_mask = tox21_df["canonical_smiles"].isin(train_smiles_set)
    n_overlap = overlap_mask.sum()
    print(f"\nCompounds overlapping with Karim training set: {n_overlap} ({100*n_overlap/len(tox21_df):.1f}%)")
    tox21_ext = tox21_df[~overlap_mask].reset_index(drop=True)
    print(f"External validation set after removing overlap: {len(tox21_ext)} compounds")
    print(tox21_ext["label"].value_counts())

    # Build graphs
    graphs = []
    for smi, label in zip(tox21_ext["canonical_smiles"], tox21_ext["label"]):
        g = graph_mod.mol_to_graph(smi, label)
        if g is not None:
            graphs.append(g)
    print(f"\nBuilt {len(graphs)} external validation graphs")

    # Load trained model
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    model = LightweightGAT(node_dim=node_dim, edge_dim=edge_dim,
                            hidden_dim=64, heads=4, num_layers=3, dropout=0.2).to(device)
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_gat_model.pt", map_location=device))
    model.eval()

    loader = DataLoader(graphs, batch_size=64)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.numpy())
            all_labels.append(batch.y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(labels, probs),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "n_compounds": len(labels),
        "n_overlap_removed": int(n_overlap),
    }

    print("\n" + "=" * 55)
    print("EXTERNAL VALIDATION RESULTS (Tox21, independent source)")
    print("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")

    import json
    with open(f"{OUT_DIR}/external_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/external_validation_metrics.json")


if __name__ == "__main__":
    main()