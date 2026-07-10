"""
Step 7-8 of workflow: Learn toxic vs non-toxic (train), evaluate.
Uses scaffold splitting (not random) -- standard practice in molecular ML so that
the test set contains structurally distinct scaffolds from training, giving a
realistic estimate of generalization to *new* drugs (which is your stated goal).
"""
import sys
sys.path.insert(0, "/home/claude/project/src")

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

from model import LightweightGAT

OUT_DIR = "/home/claude/project/outputs"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    except Exception:
        return smiles


def scaffold_split(graphs, frac_train=0.8, frac_val=0.1):
    scaffold_to_idx = defaultdict(list)
    for i, g in enumerate(graphs):
        scaffold_to_idx[get_scaffold(g.smiles)].append(i)

    scaffold_groups = list(scaffold_to_idx.values())
    random.shuffle(scaffold_groups)
    # put biggest groups first isn't necessary; random order of scaffold groups is standard

    n_total = len(graphs)
    n_train_target = int(frac_train * n_total)
    n_val_target = int(frac_val * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) < n_train_target:
            train_idx += group
        elif len(val_idx) < n_val_target:
            val_idx += group
        else:
            test_idx += group

    return train_idx, val_idx, test_idx


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(labels, probs),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }, probs, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    graphs = torch.load(f"{OUT_DIR}/molecular_graphs.pt", weights_only=False)
    print(f"Loaded {len(graphs)} graphs")

    train_idx, val_idx, test_idx = scaffold_split(graphs)
    print(f"Scaffold split -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]
    model = LightweightGAT(node_dim=node_dim, edge_dim=edge_dim,
                            hidden_dim=64, heads=4, num_layers=3, dropout=0.2).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (lightweight)")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = 0
    best_state = None
    patience, patience_counter = 10, 0
    n_epochs = 45

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_set)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_auc {val_metrics['auc']:.4f} | val_acc {val_metrics['acc']:.4f} | "
                  f"val_f1 {val_metrics['f1']:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_metrics, test_probs, test_labels = evaluate(model, test_loader, device)

    print("\n" + "=" * 50)
    print("FINAL TEST SET RESULTS (scaffold split -- held-out structures)")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    torch.save(best_state, f"{OUT_DIR}/best_gat_model.pt")
    np.save(f"{OUT_DIR}/test_probs.npy", test_probs)
    np.save(f"{OUT_DIR}/test_labels.npy", test_labels)

    import json
    with open(f"{OUT_DIR}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nSaved model to {OUT_DIR}/best_gat_model.pt")


if __name__ == "__main__":
    main()