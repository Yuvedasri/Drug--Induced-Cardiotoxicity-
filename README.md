# Causal Graph Attention Network for hERG Cardiotoxicity Prediction

A complete machine learning pipeline for predicting drug-induced cardiotoxicity (hERG channel blocking) using Graph Attention Networks with causal interpretation.

## Project Structure

```
├── data/
│   └── hERG_Karim_Dataset.xlsx          # Primary training dataset
├── outputs/                               # Generated results
│   ├── clean_herg_dataset.csv            # Cleaned data
│   ├── molecular_graphs.pt               # Graph representations
│   ├── best_gat_model.pt                 # Trained model weights
│   ├── test_metrics.json                 # Test set performance
│   ├── causal_summary.json               # Causal validation results
│   └── prediction_*.png                  # Molecule visualizations
├── model.py                               # LightweightGAT architecture
├── 01_data_prep.py                        # Data cleaning
├── 02_graph_construction.py              # SMILES → molecular graphs
├── 03_train.py                            # Model training
├── 04_causal_analysis.py                 # Causal validation
├── 05_external_validation.py             # External Tox21 validation
├── 06_predict_new_drug.py                # Prediction for new molecules
└── example_batch_predict.py              # Batch prediction example
```

## Installation

```bash
pip install rdkit torch torch_geometric pandas numpy scikit-learn scipy openpyxl
```

## Usage

### 1. Complete Pipeline (from scratch)

Run scripts in order:

```bash
python 01_data_prep.py              # Clean dataset
python 02_graph_construction.py     # Build molecular graphs
python 03_train.py                  # Train GAT model (~15-20 min on CPU)
python 04_causal_analysis.py        # Validate causal substructures
python 05_external_validation.py    # External validation (requires Tox21 data)
```

### 2. Predict New Molecule

**Command-line:**
```bash
python 06_predict_new_drug.py "CCN(CC)CCOC(=O)c1ccc(N)cc1"
```

**Interactive:**
```bash
python 06_predict_new_drug.py
# Prompts for SMILES input
```

**Programmatic (batch):**
```python
from importlib import import_module
predict_mod = import_module("06_predict_new_drug")
predict = predict_mod.predict

result = predict("c1ccccc1")
print(f"Probability: {result['probability']}")
print(f"Classification: {result['classification']}")
```

See `example_batch_predict.py` for a complete example.

## Results

### Model Performance (Test Set - Scaffold Split)
- **AUC:** 0.849
- **Accuracy:** 76.7%
- **F1 Score:** 0.761
- **Dataset:** 13,445 molecules (balanced)

### Causal Validation
- **Attention-guided masking** drops probability by 26.3%
- **Random masking** drops probability by 17.0%
- **Necessity gap:** 9.3% (p < 2.7×10⁻²¹) - highly significant
- **Chemistry validation:** 97.5% flagged atoms in aromatic rings, 42.5% basic nitrogen
- Matches known hERG pharmacophore features

## Architecture

**LightweightGAT:**
- 3-layer GATv2Conv
- 64 hidden dimensions, 4 attention heads
- Mean + max pooling
- 326,017 parameters
- Node features: 41-dim (atom properties)
- Edge features: 7-dim (bond properties)

## Key Features

✅ **Scaffold splitting** - ensures structural diversity in test set  
✅ **Causal interpretation** - identifies toxic substructures via attention  
✅ **Necessity testing** - validates causal claims with masking experiments  
✅ **Visualization** - highlights causal atoms in molecule images  
✅ **Batch prediction** - programmatic API for multiple molecules  
✅ **Error handling** - graceful failures with helpful messages  

## Example Predictions

| Molecule | Probability | Classification | Known Status |
|----------|-------------|----------------|--------------|
| Terfenadine | 0.88 | TOXIC | ✓ Known hERG blocker |
| Diphenhydramine | 0.95 | TOXIC | ✓ Known blocker |
| Propranolol | 0.66 | TOXIC | ✓ Moderate blocker |
| Aspirin | 0.01 | NON-TOXIC | ✓ Not a blocker |
| Caffeine | 0.01 | NON-TOXIC | ✓ Not a blocker |

## Output Format

```
============================================================
CARDIOTOXICITY PREDICTION
============================================================
Input SMILES: CN1CCC(C(C2=CC=CC=C2)C3=CC=CC=C3)CC1

Predicted toxicity probability: 0.8399
Classification: TOXIC (hERG blocker risk) (high confidence)

Causal substructure driving this prediction:
  - Atoms: Atom 1 (N), Atom 0 (C), Atom 3 (C), Atom 5 (C), Atom 7 (C)
  - Masking these atoms drops toxicity probability by 0.5122
  - Random-atom masking baseline drops probability by 0.0507
  - Causal necessity gap: 0.4615

Visualization saved to: outputs/prediction_20260710_111124_389.png
============================================================
```

## References

- **Dataset:** Karim et al. hERG dataset (13,445 molecules)
- **Architecture:** GATv2 (Brody et al., 2021)
- **Splitting:** Scaffold-based (preserves chemical diversity)
- **Causal validation:** Necessity testing via feature masking

## License

Research/Educational use
