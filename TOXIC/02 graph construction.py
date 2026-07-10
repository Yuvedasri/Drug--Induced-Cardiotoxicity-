"""
Step 4-6 of workflow:
  - SMILES -> RDKit creates molecule
  - Molecular graph creation: Atoms = Nodes, Bonds = Edges
Produces a list of torch_geometric.data.Data objects, one per compound.
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

OUT_DIR = "/home/claude/project/outputs"

# ---- Atom feature vocabulary ----
ATOM_LIST = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se', 'Other']
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

def one_hot(value, choices):
    vec = [0] * (len(choices) + 1)  # +1 for "other/unknown"
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1
    return vec

def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetSymbol(), ATOM_LIST[:-1])          # element type
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)    # hybridization
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])        # number of bonds
    feats += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats.append(1 if atom.GetIsAromatic() else 0)
    feats.append(1 if atom.IsInRing() else 0)
    feats.append(atom.GetMass() * 0.01)  # scaled atomic mass, continuous feature
    return feats

def bond_features(bond):
    bt = bond.GetBondType()
    feats = one_hot(bt, [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
                          rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC])
    feats.append(1 if bond.GetIsConjugated() else 0)
    feats.append(1 if bond.IsInRing() else 0)
    return feats

def mol_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Node features
    x = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # Edges (undirected -> add both directions) + edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_index) == 0:
        # single-atom molecule edge case: add a self-loop so the graph isn't empty
        edge_index = [[0, 0]]
        edge_attr = [[0] * 7]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float),
        smiles=smiles,
    )
    return data

if __name__ == "__main__":
    df = pd.read_csv(f"{OUT_DIR}/clean_herg_dataset.csv")
    graphs = []
    failed = 0
    for smi, label in zip(df["canonical_smiles"], df["label"]):
        g = mol_to_graph(smi, label)
        if g is None:
            failed += 1
            continue
        graphs.append(g)

    print(f"Built {len(graphs)} molecular graphs ({failed} failed)")
    print(f"Node feature dim: {graphs[0].x.shape[1]}")
    print(f"Edge feature dim: {graphs[0].edge_attr.shape[1]}")
    print(f"Example graph: {graphs[0]}")

    torch.save(graphs, f"{OUT_DIR}/molecular_graphs.pt")
    print(f"\nSaved {len(graphs)} graphs to {OUT_DIR}/molecular_graphs.pt")