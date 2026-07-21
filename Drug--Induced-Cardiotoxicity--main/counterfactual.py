"""
Counterfactual Validation Pipeline for hERG Cardiotoxicity Prediction.

This module implements a complete counterfactual validation workflow to verify
whether the molecular substructure identified as important by the GAT model
actually influences toxicity predictions.

Workflow:
  1. Load trained GAT model and test molecules
  2. Extract attention weights to identify important atoms
  3. Generate chemically valid counterfactual molecules
  4. Predict toxicity on counterfactual molecules
  5. Calculate probability differences and generate reports
  6. Create visualizations and statistics

Usage:
  python counterfactual.py [--num_molecules N] [--top_k K] [--num_variants V]
  
Example:
  python counterfactual.py --num_molecules 10 --top_k 0.25 --num_variants 5
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from model import LightweightGAT
from counterfactual_generator import CounterfactualGenerator
from counterfactual_visualization import CounterfactualVisualizer

OUT_DIR = "outputs"
COUNTERFACTUAL_DIR = "counterfactual_examples"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


class CounterfactualValidator:
    """
    Main orchestrator for counterfactual validation analysis.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize validator with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Torch device for inference
        """
        self.device = device if device is not None else torch.device("cpu")
        self.model = None
        self.model_path = model_path or f"{OUT_DIR}/best_gat_model.pt"
        
    def load_model(self, node_dim, edge_dim):
        """Load trained GAT model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = LightweightGAT(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            heads=4,
            num_layers=3,
            dropout=0.2
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"Loaded model from {self.model_path}")
    
    def get_node_attention(self, data):
        """
        Extract per-node attention importance scores from GAT.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            np.array: Importance scores for each node
        """
        data = data.to(self.device)
        with torch.no_grad():
            _, attn_layers = self.model(
                data.x, data.edge_index, data.edge_attr,
                torch.zeros(data.x.size(0), dtype=torch.long, device=self.device),
                return_attention=True
            )
        
        n_nodes = data.x.size(0)
        node_score = torch.zeros(n_nodes, device=self.device)
        
        # Aggregate attention across layers
        for edge_index, alpha in attn_layers:
            alpha_mean = alpha.mean(dim=1)  # Average across attention heads
            node_score.index_add_(0, edge_index[1], alpha_mean)
        
        node_score = node_score / len(attn_layers)
        return node_score.cpu().numpy()
    
    def predict(self, data):
        """
        Predict toxicity probability for a molecule.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            float: Predicted toxicity probability (0-1)
        """
        data = data.to(self.device)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, data.edge_attr, batch)
            prob = torch.sigmoid(logits).item()
        
        return prob
    
    def extract_important_atoms(self, data, top_k=0.25):
        """
        Extract important atoms from attention weights.
        
        Args:
            data: PyTorch Geometric Data object
            top_k: Fraction or absolute number of top atoms to extract
            
        Returns:
            np.array: Indices of important atoms
        """
        scores = self.get_node_attention(data)
        n_atoms = len(scores)
        
        if 0 < top_k < 1:
            k = max(1, int(np.ceil(top_k * n_atoms)))
        else:
            k = int(top_k)
        
        k = min(k, n_atoms)
        important_idxs = np.argsort(-scores)[:k]
        
        return important_idxs, scores
    
    def analyze_molecule(self, data, top_k=0.25, num_variants=5):
        """
        Perform complete counterfactual analysis for a single molecule.
        
        Args:
            data: PyTorch Geometric Data object with .smiles attribute
            top_k: Fraction of top atoms to identify as important
            num_variants: Number of counterfactual variants to generate
            
        Returns:
            dict: Analysis results
        """
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {
                'smiles': smiles,
                'error': 'Invalid SMILES',
                'valid': False
            }
        
        # Original prediction
        original_prob = self.predict(data)
        original_class = 1 if original_prob >= 0.5 else 0
        
        # Extract important atoms
        important_atoms, importance_scores = self.extract_important_atoms(data, top_k)
        
        # Generate counterfactual molecules
        counterfactuals = CounterfactualGenerator.generate_counterfactuals(
            smiles, important_atoms, num_variants=num_variants
        )
        
        # Analyze each counterfactual
        cf_results = []
        for cf_data in counterfactuals:
            cf_smiles = cf_data['smiles']
            
            # Try to convert to graph  
            try:
                from importlib import import_module
                graph_mod = import_module("02_graph_construction")
                cf_mol_data = graph_mod.mol_to_graph(cf_smiles, label=0)
                
                if cf_mol_data is None:
                    continue
                
                # Predict on counterfactual
                cf_prob = self.predict(cf_mol_data)
                cf_class = 1 if cf_prob >= 0.5 else 0
                
                # Calculate drop
                prob_drop = original_prob - cf_prob
                class_changed = (original_class != cf_class)
                
                cf_results.append({
                    'counterfactual_smiles': cf_smiles,
                    'modification_type': cf_data['modification_type'],
                    'modified_atoms': cf_data['modified_atoms'],
                    'cf_probability': cf_prob,
                    'prob_drop': prob_drop,
                    'class_changed': class_changed,
                    'cf_class': cf_class,
                    'cf_valid': True
                })
            except Exception as e:
                print(f"  Error processing counterfactual: {str(e)}")
                continue
        
        # Compile results
        result = {
            'smiles': smiles,
            'valid': True,
            'n_atoms': mol.GetNumAtoms(),
            'n_important_atoms': len(important_atoms),
            'important_atom_indices': list(important_atoms),
            'importance_scores': {int(idx): float(importance_scores[idx]) 
                                 for idx in important_atoms},
            'original_probability': float(original_prob),
            'original_class': int(original_class),
            'n_counterfactuals': len(cf_results),
            'counterfactuals': cf_results,
        }
        
        # Calculate statistics
        if len(cf_results) > 0:
            prob_drops = [cf['prob_drop'] for cf in cf_results]
            result['mean_prob_drop'] = float(np.mean(prob_drops))
            result['median_prob_drop'] = float(np.median(prob_drops))
            result['max_prob_drop'] = float(np.max(prob_drops))
            result['min_prob_drop'] = float(np.min(prob_drops))
            result['n_class_changes'] = sum(1 for cf in cf_results if cf['class_changed'])
        
        return result
    
    def visualize_result(self, result, output_dir=COUNTERFACTUAL_DIR):
        """
        Generate visualizations for analysis result.
        
        Args:
            result: Analysis result dict
            output_dir: Directory to save visualizations
            
        Returns:
            list: Paths to generated images
        """
        if not result.get('valid', False):
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        # Sanitize filename
        smiles_clean = result['smiles'].replace('/', '_').replace('\\', '_')[:50]
        base_name = f"{smiles_clean}_{int(result['original_probability']*100)}"
        
        # Generate side-by-side visualizations for top counterfactuals
        counterfactuals = result.get('counterfactuals', [])
        
        for cf_idx, cf_data in enumerate(counterfactuals[:3]):  # Top 3
            output_path = os.path.join(
                output_dir,
                f"{base_name}_cf{cf_idx}.png"
            )
            
            success = CounterfactualVisualizer.draw_side_by_side(
                original_smiles=result['smiles'],
                counterfactual_smiles=cf_data['counterfactual_smiles'],
                important_atoms=result['important_atom_indices'],
                modified_atoms=cf_data['modified_atoms'],
                output_path=output_path,
                title=f"{base_name} - {cf_data['modification_type']}\n"
                      f"Δprob: {cf_data['prob_drop']:.3f}"
            )
            
            if success:
                generated_files.append(output_path)
        
        return generated_files


def load_test_molecules(graphs_path=f"{OUT_DIR}/molecular_graphs.pt"):
    """Load molecular graphs and return test subset."""
    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"Graphs not found: {graphs_path}")
    
    # Load graphs checkpoint (explicitly set weights_only=False for compatibility)
    try:
        graphs = torch.load(graphs_path, weights_only=False, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load graphs from {graphs_path}: {e}")
    
    # Recreate test split using same logic as training
    try:
        from importlib import import_module
        train_mod = import_module("03_train")
        train_idx, val_idx, test_idx = train_mod.scaffold_split(graphs)
        test_graphs = [graphs[i] for i in test_idx]
        return test_graphs
    except Exception as e:
        print(f"Warning: Could not load test split, using full set: {e}")
        return graphs


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Counterfactual Validation for Cardiotoxicity Prediction"
    )
    parser.add_argument('--num_molecules', type=int, default=50,
                       help='Number of molecules to analyze')
    parser.add_argument('--top_k', type=float, default=0.25,
                       help='Top-K fraction of atoms to identify as important')
    parser.add_argument('--num_variants', type=int, default=5,
                       help='Number of counterfactual variants per molecule')
    parser.add_argument('--output_dir', type=str, default=COUNTERFACTUAL_DIR,
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("COUNTERFACTUAL VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load molecules
    print("Loading test molecules...")
    try:
        test_molecules = load_test_molecules()
        print(f"Loaded {len(test_molecules)} test molecules")
    except Exception as e:
        print(f"Error loading test molecules: {e}")
        return
    
    # Select subset
    n_analyze = min(args.num_molecules, len(test_molecules))
    selected_indices = np.random.choice(len(test_molecules), n_analyze, replace=False)
    test_molecules = [test_molecules[i] for i in selected_indices]
    print(f"Analyzing {n_analyze} molecules")
    print()
    
    # Initialize validator
    print("Initializing validator...")
    validator = CounterfactualValidator(device=device)
    
    # Load model
    node_dim = test_molecules[0].x.shape[1]
    edge_dim = test_molecules[0].edge_attr.shape[1]
    
    try:
        validator.load_model(node_dim, edge_dim)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print()
    
    # Analyze each molecule
    results = []
    print("Analyzing molecules...")
    print("-" * 80)
    
    for idx, data in enumerate(test_molecules):
        try:
            print(f"[{idx+1}/{n_analyze}] Analyzing: {data.smiles[:50]}...")
            
            result = validator.analyze_molecule(
                data,
                top_k=args.top_k,
                num_variants=args.num_variants
            )
            
            if result.get('valid', False):
                results.append(result)
                
                # Generate visualization
                try:
                    validator.visualize_result(result, args.output_dir)
                except Exception as e:
                    print(f"  Visualization error: {e}")
                
                # Print summary
                print(f"  ✓ Original prob: {result['original_probability']:.3f}")
                print(f"  ✓ Counterfactuals: {result['n_counterfactuals']}")
                if result['n_counterfactuals'] > 0:
                    print(f"  ✓ Mean prob drop: {result['mean_prob_drop']:.3f}")
                    print(f"  ✓ Max prob drop: {result['max_prob_drop']:.3f}")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
            continue
    
    print()
    print("=" * 80)
    print(f"ANALYSIS COMPLETE: {len(results)} molecules analyzed")
    print("=" * 80)
    print()
    
    if len(results) == 0:
        print("No valid results generated.")
        return
    
    # Generate report CSV
    report_data = []
    for result in results:
        for cf in result.get('counterfactuals', []):
            report_data.append({
                'original_smiles': result['smiles'],
                'counterfactual_smiles': cf['counterfactual_smiles'],
                'n_atoms': result['n_atoms'],
                'n_important_atoms': result['n_important_atoms'],
                'important_atom_indices': ','.join(map(str, result['important_atom_indices'])),
                'modified_atoms': ','.join(map(str, cf['modified_atoms'])),
                'modification_type': cf['modification_type'],
                'original_probability': result['original_probability'],
                'original_class': result['original_class'],
                'counterfactual_probability': cf['cf_probability'],
                'counterfactual_class': cf['cf_class'],
                'probability_drop': cf['prob_drop'],
                'class_changed': cf['class_changed'],
                'validity': cf['cf_valid']
            })
    
    if len(report_data) > 0:
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(OUT_DIR, 'counterfactual_report.csv')
        report_df.to_csv(report_path, index=False)
        print(f"Report saved: {report_path}")
        print(f"  Rows: {len(report_df)}")
        print()
    
    # Generate summary statistics
    print("STATISTICS")
    print("-" * 80)
    
    all_prob_drops = []
    all_class_changes = 0
    n_counterfactuals = 0
    
    for result in results:
        if result['n_counterfactuals'] > 0:
            all_prob_drops.extend([cf['prob_drop'] for cf in result['counterfactuals']])
            all_class_changes += result.get('n_class_changes', 0)
            n_counterfactuals += result['n_counterfactuals']
    
    if len(all_prob_drops) > 0:
        print(f"Total counterfactuals generated: {n_counterfactuals}")
        print(f"Mean probability drop: {np.mean(all_prob_drops):.4f}")
        print(f"Median probability drop: {np.median(all_prob_drops):.4f}")
        print(f"Std dev probability drop: {np.std(all_prob_drops):.4f}")
        print(f"Max probability drop: {np.max(all_prob_drops):.4f}")
        print(f"Min probability drop: {np.min(all_prob_drops):.4f}")
        print(f"Predictions changed: {all_class_changes}/{n_counterfactuals} "
              f"({100*all_class_changes/n_counterfactuals:.1f}%)")
        print()
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_molecules_analyzed': len(results),
        'n_counterfactuals_total': n_counterfactuals,
        'mean_prob_drop': float(np.mean(all_prob_drops)) if len(all_prob_drops) > 0 else 0,
        'median_prob_drop': float(np.median(all_prob_drops)) if len(all_prob_drops) > 0 else 0,
        'max_prob_drop': float(np.max(all_prob_drops)) if len(all_prob_drops) > 0 else 0,
        'min_prob_drop': float(np.min(all_prob_drops)) if len(all_prob_drops) > 0 else 0,
        'n_class_changes': int(all_class_changes),
        'pct_class_changes': float(100*all_class_changes/n_counterfactuals) if n_counterfactuals > 0 else 0,
        'parameters': {
            'top_k': args.top_k,
            'num_variants': args.num_variants,
            'num_molecules': n_analyze
        }
    }
    
    summary_path = os.path.join(OUT_DIR, 'counterfactual_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved: {summary_path}")
    print()
    
    print("✓ Counterfactual validation complete!")


if __name__ == "__main__":
    main()
