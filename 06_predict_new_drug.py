"""
Step 6: Predict cardiotoxicity (hERG blocking) for a new drug candidate.
Accepts either a drug name (e.g. 'Aspirin') or a SMILES string.
When a drug name is given, all matching SMILES are fetched from PubChem
and each is analysed with the trained Causal-GAT.

Usage:
  python 06_predict_new_drug.py "Aspirin"
  python 06_predict_new_drug.py "CCN(CC)CCOC(=O)c1ccc(N)cc1"
  python 06_predict_new_drug.py   # interactive prompt
"""
import sys
import os
import json
import urllib.request
import urllib.parse
import torch
import numpy as np
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Always resolve paths relative to this script's directory
# so the script works from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model import LightweightGAT
from importlib import import_module

OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


# ---------------------------------------------------------------------------
# Drug-name helpers
# ---------------------------------------------------------------------------

def is_smiles(text: str) -> bool:
    """Return True if text looks like a SMILES string rather than a drug name."""
    smiles_indicators = set('=#@+\\/%[]()0123456789')
    return any(c in smiles_indicators for c in text)


def get_smiles_from_drug_name(drug_name: str):
    """
    Query PubChem for all compounds matching drug_name.
    Returns (list_of_compounds, error_string).
    Each compound is a dict with keys: cid, smiles, iupac, formula, mw.
    """
    try:
        encoded = urllib.parse.quote(drug_name.strip())
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/property/IsomericSMILES,CanonicalSMILES,IUPACName,"
            f"MolecularFormula,MolecularWeight/JSON"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "CardiotoxicityPredictor/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        props = data.get("PropertyTable", {}).get("Properties", [])
        compounds = []
        for p in props[:5]:          # limit to 5 results
            # PubChem returns the key as "SMILES" (connectivity) or
            # "IsomericSMILES" depending on the request — try all variants
            smiles = (p.get("IsomericSMILES", "")
                      or p.get("CanonicalSMILES", "")
                      or p.get("SMILES", "")
                      or p.get("ConnectivitySMILES", ""))
            compounds.append({
                "cid":     p.get("CID", "?"),
                "smiles":  smiles,
                "iupac":   p.get("IUPACName", "N/A"),
                "formula": p.get("MolecularFormula", "N/A"),
                "mw":      p.get("MolecularWeight", "N/A"),
            })
        return compounds, None

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return [], f"Drug '{drug_name}' not found on PubChem."
        return [], f"PubChem HTTP error {e.code}: {e.reason}"
    except Exception as e:
        return [], f"PubChem lookup failed: {str(e)}"


# ---------------------------------------------------------------------------


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

def print_results(result: dict, label: str = None):
    """Print prediction results in a clean, readable format."""
    print("\n" + "=" * 60)
    if label:
        print(f"CARDIOTOXICITY PREDICTION -- {label}")
    else:
        print("CARDIOTOXICITY PREDICTION")
    print("=" * 60)
    print(f"Input SMILES: {result['smiles']}")

    if not result['valid'] or result['error']:
        print(f"\nERROR: {result['error']}")
        print("=" * 60)
        return

    icon = "[TOXIC]" if result['probability'] >= 0.5 else "[SAFE] "
    print(f"\nPredicted toxicity probability: {result['probability']:.4f}")
    print(f"Classification: {icon} {result['classification']} ({result['confidence']})")

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


def print_summary_table(drug_results: list):
    """Print a clean summary table for multiple drugs."""
    print("\n" + "#" * 60)
    print("  INDIVIDUAL DRUG SUMMARY")
    print("#" * 60)
    print(f"  {'Drug':<22} {'Probability':>12}  {'Result':<30}")
    print(f"  {'-'*22} {'-'*12}  {'-'*30}")
    for entry in drug_results:
        name = entry['name'][:22]
        prob = entry.get('probability')
        if prob is None:
            status = "ERROR / Not found"
            prob_str = "  N/A"
        else:
            icon = "[TOXIC]" if prob >= 0.5 else "[SAFE] "
            clf  = "TOXIC" if prob >= 0.5 else "NON-TOXIC"
            conf = entry.get('confidence', '')
            status = f"{icon} {clf} ({conf})"
            prob_str = f"{prob:>12.4f}"
        print(f"  {name:<22} {prob_str}  {status}")
    print("#" * 60 + "\n")


def analyze_combination(smiles_a: str, smiles_b: str) -> dict:
    """
    Predict toxicity of a two-drug combination.
    Uses SMILES dot-notation (mol1.mol2) which represents a disconnected
    molecular complex -- the GAT processes both fragments together.
    Returns a dict like predict(), plus 'combo_smiles'.
    """
    combo_smiles = f"{smiles_a}.{smiles_b}"
    result = predict(combo_smiles)
    result['combo_smiles'] = combo_smiles
    return result


def print_combination_matrix(drug_results: list):
    """
    For every pair of drugs, run combination analysis and display
    a risk matrix with synergy flags.
    """
    valid = [d for d in drug_results if d.get('probability') is not None and d.get('smiles')]
    if len(valid) < 2:
        print("\n[Combination analysis requires at least 2 valid drugs]\n")
        return

    print("\n" + "#" * 60)
    print("  DRUG COMBINATION ANALYSIS")
    print("#" * 60)
    print("  (Combination SMILES = Drug A . Drug B processed together)")
    print()

    pairs = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            pairs.append((valid[i], valid[j]))

    any_synergy = False
    for drug_a, drug_b in pairs:
        name_a = drug_a['name']
        name_b = drug_b['name']
        prob_a = drug_a['probability']
        prob_b = drug_b['probability']
        smiles_a = drug_a['smiles']
        smiles_b = drug_b['smiles']

        print(f"  Combination: {name_a}  +  {name_b}")
        print(f"    Individual probabilities: {name_a}={prob_a:.4f}  |  {name_b}={prob_b:.4f}")

        combo_result = analyze_combination(smiles_a, smiles_b)
        combo_prob   = combo_result.get('probability')

        if combo_prob is None:
            print(f"    Combination prediction: ERROR ({combo_result.get('error')})")
        else:
            max_individual = max(prob_a, prob_b)
            synergy = combo_prob > max_individual + 0.05   # 5% threshold
            flag = "  *** SYNERGISTIC RISK ***" if synergy else ""
            icon = "[TOXIC]" if combo_prob >= 0.5 else "[SAFE] "
            print(f"    Combination probability: {combo_prob:.4f}  {icon}{flag}")

            if synergy:
                any_synergy = True
                print(f"    >> Combined risk ({combo_prob:.4f}) exceeds highest individual"
                      f" ({max_individual:.4f}) by "
                      f"{(combo_prob - max_individual)*100:.1f}% -- potential synergistic cardiotoxicity!")
            elif combo_prob >= 0.5:
                print(f"    >> At least one drug is a known risk; combination remains TOXIC.")
            else:
                print(f"    >> No synergistic risk detected.")
        print()

    if any_synergy:
        print("  [!] WARNING: Synergistic cardiotoxicity detected in one or more pairs.")
    else:
        print("  [OK] No synergistic combination risk detected.")
    print("#" * 60 + "\n")


def resolve_drug(name_or_smiles: str) -> dict:
    """
    Resolve a single drug name or SMILES to its first PubChem compound.
    Returns dict with keys: name, smiles, probability, confidence, error.
    """
    entry = {'name': name_or_smiles.strip(), 'smiles': None,
             'probability': None, 'confidence': None, 'error': None}

    raw = name_or_smiles.strip()
    if is_smiles(raw):
        smiles = raw
    else:
        compounds, err = get_smiles_from_drug_name(raw)
        if err or not compounds:
            entry['error'] = err or "Not found on PubChem"
            return entry
        smiles = compounds[0]['smiles']
        if not smiles:
            entry['error'] = "PubChem returned no SMILES"
            return entry

    entry['smiles'] = smiles
    result = predict(smiles)

    if result.get('error'):
        entry['error'] = result['error']
    else:
        entry['probability']  = result['probability']
        entry['confidence']   = result['confidence']
        entry['classification'] = result['classification']
        entry['causal_atoms'] = result.get('causal_atoms')
        entry['causal_drop']  = result.get('causal_drop')
        entry['random_drop']  = result.get('random_drop')
        entry['visualization_path'] = result.get('visualization_path')

    return entry


def main():
    """Main entry point for command-line usage."""
    # ------------------------------------------------------------------
    # Get input
    # ------------------------------------------------------------------
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:]).strip()
    else:
        print("\n" + "=" * 60)
        print(" CARDIOTOXICITY PREDICTOR")
        print("=" * 60)
        print("Enter one drug name OR multiple drugs separated by commas:")
        print("  Single  : Aspirin")
        print("  Multiple: Aspirin, Terfenadine, Caffeine")
        print("  SMILES  : CC(=O)Oc1ccccc1C(=O)O")
        user_input = input("> ").strip()
        if not user_input:
            print("ERROR: No input provided.")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Detect single vs multiple drug input
    # ------------------------------------------------------------------
    # Split on commas; but if it looks like a SMILES treat as single input
    parts = [p.strip() for p in user_input.split(',') if p.strip()]

    # If only one token OR looks like SMILES, use single-drug flow
    if len(parts) == 1 or is_smiles(user_input):
        single = parts[0] if parts else user_input
        if is_smiles(single):
            print(f"\nDetected SMILES input. Running prediction...")
            result = predict(single)
            print_results(result)
            sys.exit(0 if (result['valid'] and not result['error']) else 1)
        else:
            print(f"\nSearching PubChem for drug: '{single}' ...")
            compounds, error = get_smiles_from_drug_name(single)
            if error:
                print(f"\nERROR: {error}")
                sys.exit(1)
            if not compounds:
                print(f"\nNo compounds found for '{single}'.")
                sys.exit(1)
            print(f"Found {len(compounds)} compound(s) on PubChem.\n")
            for i, compound in enumerate(compounds, 1):
                print(f"{'-'*60}")
                print(f"Compound {i}/{len(compounds)}")
                print(f"  PubChem CID : {compound['cid']}")
                print(f"  Formula     : {compound['formula']}  (MW: {compound['mw']} g/mol)")
                print(f"  IUPAC Name  : {compound['iupac']}")
                print(f"  SMILES      : {compound['smiles']}")
                if not compound['smiles']:
                    print("  [Skipped - no SMILES available]")
                    continue
                result = predict(compound['smiles'])
                print_results(result, label=single)
            sys.exit(0)

    # ------------------------------------------------------------------
    # MULTI-DRUG MODE
    # ------------------------------------------------------------------
    print(f"\nMulti-drug mode: {len(parts)} drug(s) detected: {', '.join(parts)}")
    print("=" * 60)

    drug_results = []
    for i, drug_name in enumerate(parts, 1):
        print(f"\n[{i}/{len(parts)}] Looking up '{drug_name}' on PubChem...")
        entry = resolve_drug(drug_name)

        if entry['error']:
            print(f"  ERROR: {entry['error']}")
        else:
            icon = "[TOXIC]" if entry['probability'] >= 0.5 else "[SAFE] "
            print(f"  SMILES      : {entry['smiles']}")
            print(f"  Probability : {entry['probability']:.4f}")
            print(f"  Result      : {icon} {entry['classification']} ({entry['confidence']})")
            if entry.get('visualization_path'):
                print(f"  Visual      : {entry['visualization_path']}")

        drug_results.append(entry)

    # Individual summary table
    print_summary_table(drug_results)

    # Combination analysis
    print_combination_matrix(drug_results)

    sys.exit(0)


if __name__ == "__main__":
    main()
