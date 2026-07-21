# Counterfactual Validation Pipeline

## Overview

This counterfactual validation pipeline implements a robust method to verify whether the molecular substructure identified as important by the Graph Attention Network (GAT) actually influences cardiotoxicity predictions.

**Key Innovation**: Instead of simply removing atoms, we generate **chemically valid counterfactual molecules** by replacing important substructures with alternative chemical moieties. This produces more meaningful explanations of model predictions.

## Methodology

### Step 1: Extract Important Atoms

- Run the trained GATv2 model on test molecules
- Extract attention weights from each layer
- Aggregate to compute per-node importance scores
- Select Top-K atoms as the "candidate toxic substructure"

### Step 2: Generate Counterfactuals

For each molecule, generate chemically valid counterfactuals by replacing important atoms with alternatives:

| Original Group                | Counterfactual          | Rationale                |
| ----------------------------- | ----------------------- | ------------------------ |
| Aromatic benzene ring (C6H5-) | Cyclohexane (-C6H11)    | Saturate aromatic system |
| Tertiary amine (N-R3)         | Secondary amine (NH-R2) | Reduce complexity        |
| Halogen (F, Cl, Br, I)        | Hydrogen (H)            | Less electronegative     |
| Nitro group (-NO2)            | Hydrogen (-H)           | Remove polar group       |
| Large alkyl chain             | Smaller alkyl group     | Reduce hydrophobicity    |

### Step 3: Predict on Counterfactuals

- Convert each counterfactual SMILES to molecular graph
- Run predictions through GAT model
- Calculate probability difference: Δprob = P(original) - P(counterfactual)

### Step 4: Validate Results

- Larger probability drops → evidence that replaced atoms were truly responsible
- Compare against random masking control baseline
- Measure percentage of predictions that change classification

### Step 5: Generate Reports

- CSV with detailed metrics for each counterfactual
- Side-by-side visualizations
- Aggregate statistics and summary

## Installation

Ensure dependencies are installed (should already be in requirements.txt):

```bash
pip install torch torch-geometric rdkit pandas matplotlib numpy scikit-learn
```

## Usage

### Basic Usage

```bash
python counterfactual.py
```

This analyzes 50 molecules (default) and generates visualizations and reports.

### Advanced Usage

```bash
python counterfactual.py \
    --num_molecules 100 \
    --top_k 0.25 \
    --num_variants 5 \
    --output_dir counterfactual_examples \
    --device cpu
```

### Parameters

- `--num_molecules`: Number of test molecules to analyze (default: 50)
- `--top_k`: Fraction (0-1) of top atoms to identify as important (default: 0.25 = top 25%)
  - Can also be an absolute number: `--top_k 5` for exactly 5 atoms
- `--num_variants`: Number of counterfactual variants to generate per molecule (default: 5)
- `--output_dir`: Directory for saving visualizations (default: counterfactual_examples)
- `--device`: Compute device - 'cpu' or 'cuda' (default: 'cpu')

### Example Command

Analyze 20 molecules with top 30% of atoms:

```bash
python counterfactual.py --num_molecules 20 --top_k 0.30 --num_variants 3
```

## Outputs

### CSV Report: `outputs/counterfactual_report.csv`

Detailed metrics for each counterfactual:

| Column                       | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| `original_smiles`            | Original molecule SMILES                              |
| `counterfactual_smiles`      | Modified molecule SMILES                              |
| `n_atoms`                    | Number of atoms in original                           |
| `n_important_atoms`          | Number of important atoms identified                  |
| `important_atom_indices`     | Comma-separated atom indices                          |
| `modified_atoms`             | Comma-separated modified atom indices                 |
| `modification_type`          | Type of modification applied                          |
| `original_probability`       | Predicted toxicity probability (0-1)                  |
| `original_class`             | Predicted class (0=non-toxic, 1=toxic)                |
| `counterfactual_probability` | Counterfactual prediction                             |
| `counterfactual_class`       | Counterfactual predicted class                        |
| `probability_drop`           | Difference in predictions (original - counterfactual) |
| `class_changed`              | Whether class prediction changed (bool)               |
| `validity`                   | Whether counterfactual is chemically valid (bool)     |

### Summary JSON: `outputs/counterfactual_summary.json`

Aggregate statistics:

```json
{
  "timestamp": "2024-07-20T10:30:45.123456",
  "n_molecules_analyzed": 50,
  "n_counterfactuals_total": 250,
  "mean_prob_drop": 0.2451,
  "median_prob_drop": 0.1823,
  "max_prob_drop": 0.8945,
  "min_prob_drop": 0.0012,
  "n_class_changes": 42,
  "pct_class_changes": 16.8,
  "parameters": {
    "top_k": 0.25,
    "num_variants": 5,
    "num_molecules": 50
  }
}
```

### Visualizations: `counterfactual_examples/*.png`

Side-by-side comparisons showing:

- **Left**: Original molecule with important atoms in RED
- **Right**: Counterfactual with modified atoms in BLUE

Filenames encode the molecule and prediction:

```
CCO_c1ccccc1_toxicity_50_cf0.png
^                        ^  ^
SMILES (first 50 chars)  |  |
                    Probability  Variant
                       (%)      Index
```

## Interpretation Guide

### Probability Drop (Δprob)

**High Δprob (> 0.2)**: Strong evidence that the replaced atoms were responsible for toxicity

- The model's prediction changed significantly when atoms were removed
- Suggests good explanation from attention mechanism

**Low Δprob (< 0.05)**: Weak evidence

- Model's prediction barely changed
- Atoms may not be critical to the prediction
- Could indicate overfitting or non-specific attention

**Medium Δprob (0.05 - 0.2)**: Moderate evidence

- Atoms contribute to prediction but aren't strictly necessary
- May indicate polypharmacology or redundant features

### Classification Changes

When `class_changed = True`, the counterfactual's predicted class flipped:

- Original: Toxic (prob > 0.5)
- Counterfactual: Non-toxic (prob < 0.5)

High percentage of classification changes → strong evidence of causality

### Modification Types

| Type                     | Meaning                           |
| ------------------------ | --------------------------------- |
| `atom_removal`           | Atoms were removed entirely       |
| `substituent_reduction`  | Substituents made smaller/simpler |
| `benzene_to_cyclohexane` | Aromatic ring saturated           |
| `pyridine_to_piperidine` | Aromatic ring saturated           |
| `halogen_to_hydrogen`    | Halogen replaced with H           |
| `nitro_to_hydrogen`      | Nitro group removed               |
| `random_removal`         | Control: random atoms removed     |

## Quality Metrics

### Chemical Validity

All counterfactuals are validated using RDKit:

- Proper valency
- Valid bond orders
- Reasonable molecular weight (100-600 Da)
- Correct atom connectivity

### Statistical Significance

To assess if attention-based modifications are more impactful than random:

```python
# Compare metrics for different groups
import pandas as pd

df = pd.read_csv('outputs/counterfactual_report.csv')

# Attention-guided modifications
attention_drops = df[df['modification_type'] != 'random_removal']['probability_drop']

# Random control
random_drops = df[df['modification_type'] == 'random_removal']['probability_drop']

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(attention_drops, random_drops)

print(f"Mean attention drop: {attention_drops.mean():.4f}")
print(f"Mean random drop: {random_drops.mean():.4f}")
print(f"t-test p-value: {p_value:.6f}")
```

## Advanced Usage

### Custom Analysis Script

```python
from counterfactual import CounterfactualValidator
from importlib import import_module
import torch

# Load molecules
graphs = torch.load('outputs/molecular_graphs.pt', weights_only=False)

# Initialize validator
validator = CounterfactualValidator(device=torch.device('cpu'))
validator.load_model(
    node_dim=graphs[0].x.shape[1],
    edge_dim=graphs[0].edge_attr.shape[1]
)

# Analyze single molecule
result = validator.analyze_molecule(
    graphs[0],
    top_k=0.25,
    num_variants=5
)

print(f"Original probability: {result['original_probability']:.3f}")
print(f"Mean probability drop: {result['mean_prob_drop']:.3f}")

# Generate visualization
validator.visualize_result(result)
```

### Filter by Toxicity

```python
import pandas as pd

df = pd.read_csv('outputs/counterfactual_report.csv')

# Only toxic predictionsfiltered = df[df['original_class'] == 1]

# High probability drops
high_impact = df[df['probability_drop'] > 0.3]

print(f"Molecules with high-impact modifications: {len(high_impact)}")
```

### Generate Statistics

```python
import numpy as np

df = pd.read_csv('outputs/counterfactual_report.csv')

print("PROBABILITY DROP STATISTICS")
print(f"Mean: {df['probability_drop'].mean():.4f}")
print(f"Median: {df['probability_drop'].median():.4f}")
print(f"Std Dev: {df['probability_drop'].std():.4f}")
print(f"Q1: {df['probability_drop'].quantile(0.25):.4f}")
print(f"Q3: {df['probability_drop'].quantile(0.75):.4f}")

print("\nCLASSIFICATION CHANGES")
n_changes = df['class_changed'].sum()
total = len(df)
print(f"{n_changes}/{total} predictions changed ({100*n_changes/total:.1f}%)")

print("\nMODIFICATION TYPE DISTRIBUTION")
print(df['modification_type'].value_counts())
```

## Troubleshooting

### Issue: "Model not found"

**Solution**: Run `03_train.py` first to train the model

### Issue: "Invalid SMILES" errors

**Solution**: Check that `02_graph_construction.py` can parse the molecules

### Issue: Slow execution

**Solution**:

- Reduce `--num_molecules`
- Use `--device cuda` if GPU available
- Reduce `--num_variants`

### Issue: Memory errors

**Solution**:

- Reduce batch size in model
- Process fewer molecules per run
- Use CPU instead of GPU

## Citation

If you use this counterfactual validation pipeline in your research, please cite:

```bibtex
@article{cardiotoxicity2024,
  title={Explainable hERG Cardiotoxicity Prediction with Graph Neural Networks and Counterfactual Validation},
  author={Your Name},
  year={2024}
}
```

## References

1. **Attention Mechanisms in GNNs**: Brody et al. "Attention is All You Need" (Vaswani et al., 2017)
2. **Graph Neural Networks**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
3. **Counterfactual Explanations**: Wachter et al., "Counterfactual Explanations without Opening the Black Box" (2019)
4. **Cheminformatics**: Landrum, "RDKit: Open-source cheminformatics" (2006-)
5. **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric" (ICLR 2019 Workshop)

## Contact

For questions or issues, please refer to the main project README.
