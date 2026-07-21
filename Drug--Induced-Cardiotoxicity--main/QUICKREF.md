# Counterfactual Validation - Quick Reference Card

## TL;DR - Start Here

### Installation

```bash
# Dependencies already in requirements.txt
pip install -r requirements.txt  # (if not done already)
```

### Run Analysis

```bash
python counterfactual.py --num_molecules 20
```

### Check Results

```bash
# CSV metrics
head -20 outputs/counterfactual_report.csv

# Summary statistics
cat outputs/counterfactual_summary.json

# Images
ls counterfactual_examples/*.png
```

### Interpret Results

- **Mean probability drop > 0.15** = good explanation (atoms matter)
- **> 10% class changes** = strong causal signal
- **All visualizations valid** = chemically sound counterfactuals

---

## One-Page Reference

### Command Examples

| Goal               | Command                                                          |
| ------------------ | ---------------------------------------------------------------- |
| **Quick test**     | `python counterfactual.py --num_molecules 5`                     |
| **Default run**    | `python counterfactual.py`                                       |
| **Thorough**       | `python counterfactual.py --num_molecules 100 --num_variants 10` |
| **High quality**   | `python counterfactual.py --top_k 0.30 --num_variants 8`         |
| **Validate setup** | `python validate_counterfactual.py`                              |

### Key Parameters

| Parameter         | Default | Range    | Effect                        |
| ----------------- | ------- | -------- | ----------------------------- |
| `--num_molecules` | 50      | 1-1000   | How many to analyze           |
| `--top_k`         | 0.25    | 0.1-0.5  | % of top atoms as "important" |
| `--num_variants`  | 5       | 1-20     | Counterfactuals per molecule  |
| `--device`        | cpu     | cpu/cuda | Compute device                |

### Output Files

| File                            | Contains                      | Format          |
| ------------------------------- | ----------------------------- | --------------- |
| `counterfactual_report.csv`     | All counterfactuals + metrics | CSV (100+ rows) |
| `counterfactual_summary.json`   | Aggregate statistics          | JSON            |
| `counterfactual_examples/*.png` | Visual comparisons            | PNG images      |

### CSV Columns (Key Ones)

| Column                 | Example                | Interpretation              |
| ---------------------- | ---------------------- | --------------------------- |
| `original_probability` | 0.823                  | How toxic (0-1)             |
| `probability_drop`     | 0.312                  | How much prediction changed |
| `class_changed`        | True                   | Did prediction flip?        |
| `modification_type`    | benzene_to_cyclohexane | What was changed            |
| `validity`             | True                   | Is it chemically valid?     |

### Typical Results

- **Molecules analyzed**: 50
- **Counterfactuals generated**: 250 (5 per molecule)
- **Mean Δprob**: 0.15-0.35 (normal range)
- **Class changes**: 10-30%
- **Execution time**: 5-15 minutes
- **Output size**: 50-200 MB

---

## Troubleshooting

| Problem                  | Solution                                                                   |
| ------------------------ | -------------------------------------------------------------------------- |
| **Module not found**     | `cd Drug--Induced-Cardiotoxicity--main/Drug--Induced-Cardiotoxicity--main` |
| **Model not found**      | Run `python 03_train.py` first                                             |
| **Slow execution**       | Reduce `--num_molecules` or `--num_variants`                               |
| **Memory errors**        | Use smaller dataset or `--device cpu`                                      |
| **PIL errors**           | `pip install pillow`                                                       |
| **DLL errors (Windows)** | Reinstall PyTorch: `pip install torch --force-reinstall`                   |

---

## Python API Quick Start

### Analyze Single Molecule

```python
from counterfactual import CounterfactualValidator
import torch

# Load validator
validator = CounterfactualValidator(device=torch.device('cpu'))
validator.load_model(node_dim=25, edge_dim=7)

# Analyze
result = validator.analyze_molecule(
    data=graph_object,
    top_k=0.25,
    num_variants=5
)

# Results
print(f"Important atoms: {result['important_atom_indices']}")
print(f"Mean Δprob: {result['mean_prob_drop']:.3f}")
```

### Generate Counterfactuals Only

```python
from counterfactual_generator import CounterfactualGenerator

smiles = "CCO"  # ethanol
important_atoms = [0, 1]

counterfactuals = CounterfactualGenerator.generate_counterfactuals(
    smiles, important_atoms, num_variants=5
)

for cf in counterfactuals:
    print(f"{cf['smiles']} - {cf['modification_type']}")
```

### Visualize Results

```python
from counterfactual_visualization import CounterfactualVisualizer

CounterfactualVisualizer.draw_side_by_side(
    original_smiles="CCO",
    counterfactual_smiles="CC",
    important_atoms=[0],
    modified_atoms=[1],
    output_path="example.png"
)
```

---

## Metrics Explained

### Probability Drop (Δprob)

```
Δprob = P(original) - P(counterfactual)

0.00 - 0.05   → Atoms not important (weak)
0.05 - 0.15   → Minor importance (moderate)
0.15 - 0.30   → Strong importance (good)
0.30 - 1.00   → Critical importance (very good)
```

### Classification Accuracy

```
% class changes = (# flipped predictions) / (total counterfactuals) × 100

0% - 5%      → Predictions very stable (robust model)
5% - 20%     → Normal range (good model)
20% - 50%    → Some instability (causal atoms found)
> 50%        → Very unstable (suggests overfitting)
```

### Validity Rate

```
% valid = (# valid counterfactuals) / (# total generated) × 100

> 90%   → Excellent (transformation strategies working)
70-90%  → Good
50-70%  → Acceptable (some failed transformations)
< 50%   → Poor (check molecule size, structure)
```

---

## File Locations

```
Your working directory
├── counterfactual.py                  ← RUN THIS
├── counterfactual_generator.py        (helper)
├── counterfactual_visualization.py    (helper)
├── COUNTERFACTUAL_README.md           (full docs)
├── QUICKSTART.md                      (this file)
├── ARCHITECTURE.md                    (technical)
├── IMPLEMENTATION_SUMMARY.md          (details)
│
├── outputs/
│   ├── best_gat_model.pt              (model)
│   ├── molecular_graphs.pt            (data)
│   ├── counterfactual_report.csv      ← MAIN RESULTS
│   └── counterfactual_summary.json    ← STATISTICS
│
└── counterfactual_examples/
    ├── molecule_50_cf0.png            ← VISUALIZATIONS
    ├── molecule_50_cf1.png
    └── ...
```

---

## Performance Tips

| Want             | Do This                                         |
| ---------------- | ----------------------------------------------- |
| **Speed**        | `--num_molecules 10 --num_variants 2`           |
| **Accuracy**     | `--num_molecules 100 --num_variants 10`         |
| **Balance**      | `--num_molecules 50 --num_variants 5` (default) |
| **GPU**          | `--device cuda` (if available)                  |
| **Reproducible** | Results always reproducible (seed=42)           |

---

## Statistical Analysis

### Compare Strategies

```python
import pandas as pd

df = pd.read_csv('outputs/counterfactual_report.csv')

# By modification type
for mod in df['modification_type'].unique():
    subset = df[df['modification_type'] == mod]
    print(f"{mod}: mean={subset['probability_drop'].mean():.4f}")
```

### Test Significance

```python
from scipy.stats import ttest_ind

attention = df[df['modification_type'] != 'random_removal']['probability_drop']
random = df[df['modification_type'] == 'random_removal']['probability_drop']

t, p = ttest_ind(attention, random)
print(f"p-value: {p:.6f}")  # < 0.05 = significant
```

### Rank Molecules

```python
# Most important atoms (largest drops)
top = df.groupby('original_smiles')['probability_drop'].mean().nlargest(10)

# Molecules with prediction flips
flipped = df[df['class_changed']].groupby('original_smiles').size().nlargest(5)
```

---

## When to Run This

### Use counterfactual validation to:

✓ **Explain predictions** - Why did model predict toxic/non-toxic?
✓ **Validate attention** - Are GAT weights actually important?
✓ **Build confidence** - Can we trust the model?
✓ **Regulatory docs** - Show explainability to regulators
✓ **Research papers** - Demonstrate XAI methodology
✓ **Identify issues** - Spot model overfitting/artifacts

### Don't use for:

✗ Real-time predictions (use `06_predict_new_drug.py` instead)
✗ Training (use `03_train.py` instead)
✗ Quick inference (too slow)

---

## Next Steps

1. **Run**: `python counterfactual.py --num_molecules 10`
2. **Wait**: ~2 minutes
3. **Inspect**: `cat outputs/counterfactual_summary.json`
4. **View**: Images in `counterfactual_examples/`
5. **Analyze**: Read `COUNTERFACTUAL_README.md`
6. **Extend**: Follow Python API examples above
7. **Scale**: Increase `--num_molecules` for production

---

## Cheat Sheet - Common Tasks

```bash
# Run quick validation
python validate_counterfactual.py

# Analyze 50 molecules (20 min)
python counterfactual.py

# High quality (60+ min)
python counterfactual.py --num_molecules 200 --num_variants 10

# View results
head -50 outputs/counterfactual_report.csv

# Count rows
wc -l outputs/counterfactual_report.csv

# List images
ls -lh counterfactual_examples/

# Copy results
cp outputs/counterfactual_* my_results/
cp -r counterfactual_examples/ my_results/
```

---

**Ready?** → Run: `python counterfactual.py --num_molecules 20` ✓
