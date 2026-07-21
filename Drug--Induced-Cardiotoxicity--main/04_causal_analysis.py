"""
Causal component of the Causal-GAT (lightweight approach):
  1. Extract GATv2 attention weights -> node-level importance per molecule.
  2. Identify top-attention "candidate causal substructure" (top-k atoms).
  3. Intervention/necessity test: mask those atoms (silence features), measure
     drop in predicted toxicity probability.
  4. Control: mask random atoms (same count, multiple trials), measure drop.
  5. Statistical test: is attention-guided masking causally more impactful
     than random masking? (paired t-test across molecules)
  6. Chemistry sanity check: do flagged atoms correspond to known hERG
     pharmacophore features (basic nitrogen, aromatic rings)?
"""
import torch
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model import LightweightGAT

OUT_DIR = "outputs"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_node_attention(model, data, device):
    """Run forward pass with attention, aggregate to a per-node importance score."""
    data = data.to(device)
    with torch.no_grad():
        _, attn_layers = model(data.x, data.edge_index, data.edge_attr,
                                torch.zeros(data.x.size(0), dtype=torch.long, device=device),
                                return_attention=True)
    n_nodes = data.x.size(0)
    node_score = torch.zeros(n_nodes, device=device)
    for edge_index, alpha in attn_layers:
        # alpha: [num_edges, heads] -> average across heads
        alpha_mean = alpha.mean(dim=1)
        # accumulate attention received by each target node
        node_score.index_add_(0, edge_index[1], alpha_mean)
    node_score = node_score / len(attn_layers)
    return node_score.cpu().numpy()


def masked_predict(model, data, mask_atom_idxs, device, clinical_features=None):
    """Predict toxicity probability with given atom indices' features zeroed out."""
    data = data.clone().to(device)
    x = data.x.clone()
    if len(mask_atom_idxs) > 0:
        x[mask_atom_idxs] = 0.0
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, data.edge_index, data.edge_attr, batch, clinical_features=clinical_features)
        prob = torch.sigmoid(logits).item()
    return prob


def flag_pharmacophore_features(mol, atom_idxs):
    """Check whether masked atoms include known hERG-relevant features:
    basic nitrogen (protonatable amine) or aromatic ring atoms."""
    has_basic_n = False
    has_aromatic = False
    for idx in atom_idxs:
        atom = mol.GetAtomWithIdx(int(idx))
        if atom.GetSymbol() == "N" and not atom.GetIsAromatic():
            # crude basic-amine heuristic: sp3 N, not amide (no adjacent C=O)
            neighbors = atom.GetNeighbors()
            is_amide = any(
                n.GetSymbol() == "C" and any(
                    b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(n).GetSymbol() == "O"
                    for b in n.GetBonds()
                ) for n in neighbors
            )
            if not is_amide:
                has_basic_n = True
        if atom.GetIsAromatic():
            has_aromatic = True
    return has_basic_n, has_aromatic


def main():
    device = torch.device("cpu")
    graphs = torch.load(f"{OUT_DIR}/molecular_graphs.pt", weights_only=False)
    node_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1]

    model = LightweightGAT(node_dim=node_dim, edge_dim=edge_dim,
                            hidden_dim=64, heads=4, num_layers=3, dropout=0.2).to(device)
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_gat_model.pt", map_location=device))
    model.eval()

    # Re-derive the same scaffold test split used in training (same seed/logic)
    from importlib import import_module
    train_mod = import_module("03_train")
    train_idx, val_idx, test_idx = train_mod.scaffold_split(graphs)
    test_graphs = [graphs[i] for i in test_idx]
    print(f"Running causal analysis on {len(test_graphs)} held-out test molecules")

    TOP_FRAC = 0.25   # top 25% attention atoms = candidate causal substructure
    N_RANDOM_TRIALS = 5

    records = []
    for gi, data in enumerate(test_graphs):
        n_atoms = data.x.size(0)
        k = max(1, int(np.ceil(TOP_FRAC * n_atoms)))

        node_scores = get_node_attention(model, data, device)
        top_idxs = np.argsort(-node_scores)[:k]

        orig_prob = masked_predict(model, data, [], device)
        attn_masked_prob = masked_predict(model, data, top_idxs, device)
        attn_drop = orig_prob - attn_masked_prob

        random_drops = []
        rng = np.random.RandomState(SEED + gi)
        for t in range(N_RANDOM_TRIALS):
            rand_idxs = rng.choice(n_atoms, size=k, replace=False)
            rand_prob = masked_predict(model, data, rand_idxs, device)
            random_drops.append(orig_prob - rand_prob)
        mean_random_drop = float(np.mean(random_drops))

        mol = Chem.MolFromSmiles(data.smiles)
        has_basic_n, has_aromatic = flag_pharmacophore_features(mol, top_idxs)

        records.append({
            "smiles": data.smiles,
            "true_label": int(data.y.item()),
            "orig_prob": orig_prob,
            "attn_drop": attn_drop,
            "random_drop": mean_random_drop,
            "necessity_gap": attn_drop - mean_random_drop,
            "n_atoms": n_atoms,
            "k_masked": k,
            "has_basic_n_in_topattn": has_basic_n,
            "has_aromatic_in_topattn": has_aromatic,
        })

        if (gi + 1) % 200 == 0:
            print(f"  processed {gi+1}/{len(test_graphs)}")

    results = pd.DataFrame(records)
    results.to_csv(f"{OUT_DIR}/causal_analysis_results.csv", index=False)

    # Restrict the core causal claim to molecules the model actually predicts as toxic --
    # masking a "toxicity substructure" should only matter when toxicity was predicted.
    toxic_pred = results[results["orig_prob"] >= 0.5]
    print(f"\nMolecules predicted toxic (orig_prob >= 0.5): {len(toxic_pred)}")

    t_stat, p_val = stats.ttest_rel(toxic_pred["attn_drop"], toxic_pred["random_drop"])

    print("\n" + "=" * 60)
    print("CAUSAL VALIDATION RESULTS (attention-guided vs random masking)")
    print("=" * 60)
    print(f"Mean prob drop | attention-guided masking : {toxic_pred['attn_drop'].mean():.4f}")
    print(f"Mean prob drop | random masking            : {toxic_pred['random_drop'].mean():.4f}")
    print(f"Mean necessity gap (attn - random)         : {toxic_pred['necessity_gap'].mean():.4f}")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.2e}")

    pct_basic_n = toxic_pred["has_basic_n_in_topattn"].mean() * 100
    pct_aromatic = toxic_pred["has_aromatic_in_topattn"].mean() * 100
    print(f"\nChemistry sanity check (top-attention substructures in predicted-toxic molecules):")
    print(f"  contain basic nitrogen : {pct_basic_n:.1f}%")
    print(f"  contain aromatic ring  : {pct_aromatic:.1f}%")
    print("  (known hERG pharmacophore = basic amine + aromatic/hydrophobic group)")

    import json
    summary = {
        "n_test_molecules": len(results),
        "n_predicted_toxic": len(toxic_pred),
        "mean_attn_drop": float(toxic_pred["attn_drop"].mean()),
        "mean_random_drop": float(toxic_pred["random_drop"].mean()),
        "mean_necessity_gap": float(toxic_pred["necessity_gap"].mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "pct_basic_n_in_top_attention": float(pct_basic_n),
        "pct_aromatic_in_top_attention": float(pct_aromatic),
    }
    with open(f"{OUT_DIR}/causal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed results to {OUT_DIR}/causal_analysis_results.csv")
    print(f"Saved summary to {OUT_DIR}/causal_summary.json")


if __name__ == "__main__":
    main()