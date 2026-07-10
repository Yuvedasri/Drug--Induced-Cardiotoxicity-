"""
Step 6: Predict cardiotoxicity (hERG blocking) for a new drug candidate.
Takes a SMILES string, runs inference with the trained Causal-GAT, identifies
the causal substructure driving the prediction, and visualizes the result.

Usage:
  python 06_predict_new_drug.py "CCN(CC)CCOC(=O)c1ccc(N)cc1"
  python 06_predict_new_drug.py   # prompts for SMILES input
"""
import sys
import os
import torch
import numpy as np
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model import LightweightGAT
from importlib import import_module

OUT_DIR = "outputs"


def predict(smiles: str) -> dict:
    """
    Predict cardiotoxicity for a given SMILES string.
    
    Args:
        smiles: SMILES representation of the molecule
        
    Returns:
        dict with keys: 'smiles', 'valid', 'probability', 'classification',
        'confidence', 'causal_atoms', 'causal_drop', 'random_drop', 
        'visualization_path', 'error'
    """
    result = {
        'smiles': smiles,
        'valid': False,
        'probability': None,
        'classification': None,
        'confidence': None,
        'causal_atoms': None,
        'causal_drop': None,
        'random_drop': None,
        'visualization_path': None,
        'error': None
    }
    
    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result['error'] = "Invalid SMILES string - RDKit cannot parse this molecule"
        return result
    
    try:
        # Sanitize molecule
        Chem.SanitizeMol(mol)
    except Exception as e:
        result['error'] = f"RDKit sanitization failed: {str(e)}"
        return result
    
    if mol.GetNumAtoms() == 0:
        result['error'] = "Molecule has no atoms"
        return result
    
    result['valid'] = True
    canonical_smiles = Chem.MolToSmiles(mol)
    
    # Check if model exists
    model_path = f"{OUT_DIR}/best_gat_model.pt"
    if not os.path.exists(model_path):
        result['error'] = f"Trained model not found at {model_path}. Please run 03_train.py first."
        return result
    
    # Import graph construction module
    try:
        graph_mod = import_module("02_graph_construction")
    except Exception as e:
        result['error'] = f"Failed to import graph construction module: {str(e)}"
        return result
    
    # Convert to graph
    data = graph_mod.mol_to_graph(canonical_smiles, label=0)  # dummy label
    if data is None:
        result['error'] = "Failed to convert molecule to graph"
        return result

    
    # Load model
    device = torch.device("cpu")
    node_dim = data.x.shape[1]
    edge_dim = data.edge_attr.shape[1]
    
    try:
        model = LightweightGAT(node_dim=node_dim, edge_dim=edge_dim,
                                hidden_dim=64, heads=4, num_layers=3, dropout=0.2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        result['error'] = f"Failed to load model: {str(e)}"
        return result
    
    # Run prediction
    data = data.to(device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, batch)
        prob = torch.sigmoid(logits).item()
    
    result['probability'] = prob
    result['classification'] = "TOXIC (hERG blocker risk)" if prob >= 0.5 else "NON-TOXIC"
    
    # Determine confidence
    if prob < 0.2 or prob > 0.8:
        result['confidence'] = "high confidence"
    elif 0.4 <= prob <= 0.6:
        result['confidence'] = "borderline"
    else:
        result['confidence'] = "moderate confidence"
    
    # Import causal analysis functions
    try:
        causal_mod = import_module("04_causal_analysis")
        get_node_attention = causal_mod.get_node_attention
        masked_predict = causal_mod.masked_predict
    except Exception as e:
        result['error'] = f"Failed to import causal analysis module: {str(e)}"
        return result
    
    # Identify causal substructure
    n_atoms = data.x.size(0)
    TOP_FRAC = 0.25
    k = max(1, int(np.ceil(TOP_FRAC * n_atoms)))
    
    try:
        node_scores = get_node_attention(model, data, device)
        top_idxs = np.argsort(-node_scores)[:k]
        
        # Calculate causal necessity
        orig_prob = masked_predict(model, data, [], device)
        attn_masked_prob = masked_predict(model, data, top_idxs, device)
        causal_drop = orig_prob - attn_masked_prob
        
        # Compare to random baseline
        rng = np.random.RandomState(42)
        random_drops = []
        for _ in range(5):
            rand_idxs = rng.choice(n_atoms, size=k, replace=False)
            rand_prob = masked_predict(model, data, rand_idxs, device)
            random_drops.append(orig_prob - rand_prob)
        random_drop = float(np.mean(random_drops))
        
        result['causal_drop'] = causal_drop
        result['random_drop'] = random_drop
        
        # Get atom details
        causal_atoms = []
        for idx in top_idxs:
            atom = mol.GetAtomWithIdx(int(idx))
            causal_atoms.append({
                'index': int(idx),
                'symbol': atom.GetSymbol(),
                'aromatic': atom.GetIsAromatic()
            })
        result['causal_atoms'] = causal_atoms
        
    except Exception as e:
        result['error'] = f"Causal analysis failed: {str(e)}"
        return result
    
    # Generate visualization
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        smiles_hash = abs(hash(canonical_smiles)) % 10000
        filename = f"prediction_{timestamp}_{smiles_hash}.png"
        filepath = os.path.join(OUT_DIR, filename)
        
        # Highlight causal atoms
        highlight_atoms = [int(idx) for idx in top_idxs]
        highlight_colors = {int(idx): (1.0, 0.6, 0.2) for idx in top_idxs}  # orange
        
        drawer = Draw.MolDraw2DCairo(800, 600)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, 
                             highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        drawer.WriteDrawingText(filepath)
        
        result['visualization_path'] = filepath
        
    except Exception as e:
        result['error'] = f"Visualization failed: {str(e)}"
        # Don't return here - visualization is optional, rest of prediction succeeded
    
    return result


def print_results(result: dict):
    """Print prediction results in a clean, readable format."""
    print("\n" + "=" * 60)
    print("CARDIOTOXICITY PREDICTION")
    print("=" * 60)
    print(f"Input SMILES: {result['smiles']}")
    
    if not result['valid'] or result['error']:
        print(f"\nERROR: {result['error']}")
        print("=" * 60)
        return
    
    print(f"\nPredicted toxicity probability: {result['probability']:.4f}")
    print(f"Classification: {result['classification']} ({result['confidence']})")
    
    if result['causal_atoms']:
        print(f"\nCausal substructure driving this prediction:")
        atom_list = ", ".join([f"Atom {a['index']} ({a['symbol']})" 
                               for a in result['causal_atoms']])
        print(f"  - Atoms: {atom_list}")
        print(f"  - Masking these atoms drops toxicity probability by {result['causal_drop']:.4f}")
        print(f"  - Random-atom masking baseline drops probability by {result['random_drop']:.4f}")
        print(f"  - Causal necessity gap: {result['causal_drop'] - result['random_drop']:.4f}")
    
    if result['visualization_path']:
        print(f"\nVisualization saved to: {result['visualization_path']}")
    
    print("=" * 60 + "\n")


def main():
    """Main entry point for command-line usage."""
    # Get SMILES from command line or prompt
    if len(sys.argv) > 1:
        smiles = sys.argv[1]
    else:
        print("Enter SMILES string for cardiotoxicity prediction:")
        smiles = input("> ").strip()
        if not smiles:
            print("ERROR: No SMILES provided")
            sys.exit(1)
    
    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Run prediction
    result = predict(smiles)
    
    # Print results
    print_results(result)
    
    # Exit with appropriate code
    if not result['valid'] or result['error']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
