# Quick Start Guide - Counterfactual Validation

## In 5 Minutes

### 1. Verify Setup

```bash
python validate_counterfactual.py
```

You should see:

```
✓ counterfactual_generator imported
✓ counterfactual_visualization imported
✓ Validator initialized
✓ Model file found
✓ VALIDATION COMPLETE
```

### 2. Run Analysis

```bash
python counterfactual.py --num_molecules 10
```

This analyzes 10 molecules and generates:

- CSV report
- JSON summary
- Visualizations

### 3. Check Results

```bash
# View report
python -c "import pandas as pd; df = pd.read_csv('outputs/counterfactual_report.csv'); print(df.head())"

# View summary
cat outputs/counterfactual_summary.json
```

---

## Understanding the Output

### Key Metrics

**Probability Drop (Δprob)**

- How much the toxicity prediction decreases when important atoms are removed
- Higher = stronger evidence that atoms cause toxicity
- Typical range: 0.05 - 0.40
- < 0.05: weak explanation
- > 0.20: strong explanation

**Classification Changes**

- Percentage of counterfactuals where predicted class flipped
- From toxic → non-toxic (or vice versa)
- Higher = more important atoms identified
- Typical range: 5% - 30%

### Example Results

```csv
original_smiles,original_probability,counterfactual_smiles,probability_drop
CCO,0.823,CC,0.311    ← Large drop = good explanation
CCN(CC)C,0.156,CC,0.012 ← Small drop = less important atoms
c1ccccc1,0.945,C1CCCC1,0.587 ← Aromatic critical to toxicity
```

---

## Command Examples

### Minimal (1 minute)

```bash
python counterfactual.py --num_molecules 5 --num_variants 2
```

### Standard (20 minutes)

```bash
python counterfactual.py --num_molecules 50 --top_k 0.25 --num_variants 5
```

### Thorough (60+ minutes)

```bash
python counterfactual.py --num_molecules 200 --top_k 0.30 --num_variants 10
```

### Only toxic predictions

```bash
# First run analysis, then filter
python counterfactual.py --num_molecules 50

# Then analyze
python -c "
import pandas as pd
df = pd.read_csv('outputs/counterfactual_report.csv')
toxic = df[df['original_class'] == 1]
print(f'Toxic molecules: {len(toxic)}')
print(f'Mean drop (toxic): {toxic[\"probability_drop\"].mean():.4f}')
"
```

---

## Parameter Tuning

### For Speed

```bash
--num_molecules 10 --num_variants 2 --top_k 0.20
```

### For Accuracy

```bash
--num_molecules 100 --num_variants 10 --top_k 0.30
```

### For Exploration

```bash
--num_molecules 50 --num_variants 5 --top_k 0.25
```

---

## Advanced Analysis

### Statistical Testing

```python
import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv('outputs/counterfactual_report.csv')

# Compare attention-guided vs random
attention = df[df['modification_type'] != 'random_removal']['probability_drop']
random = df[df['modification_type'] == 'random_removal']['probability_drop']

t_stat, p_value = ttest_ind(attention, random)
print(f"Attention mean: {attention.mean():.4f}")
print(f"Random mean: {random.mean():.4f}")
print(f"p-value: {p_value:.6f}")
```

### Filter Results

```python
import pandas as pd

df = pd.read_csv('outputs/counterfactual_report.csv')

# High impact modifications
high_impact = df[df['probability_drop'] > 0.3]
print(f"High impact: {len(high_impact)} counterfactuals")

# Prediction flips
flipped = df[df['class_changed'] == True]
print(f"Flipped: {len(flipped)} ({100*len(flipped)/len(df):.1f}%)")

# By modification type
for mod_type in df['modification_type'].unique():
    subset = df[df['modification_type'] == mod_type]
    print(f"{mod_type}: mean={subset['probability_drop'].mean():.4f}")
```

### Molecule Analysis

```python
import pandas as pd

df = pd.read_csv('outputs/counterfactual_report.csv')

# Find top molecules with biggest drops
top_molecules = df.groupby('original_smiles')['probability_drop'].mean().nlargest(10)
print("Molecules with largest average probability drops:")
print(top_molecules)
```

---

## Troubleshooting

### "Module not found" Error

**Problem**: ModuleNotFoundError when importing

**Solution**: Ensure you're in the correct directory:

```bash
cd Drug--Induced-Cardiotoxicity--main\Drug--Induced-Cardiotoxicity--main
python counterfactual.py
```

### "Model not found" Error

**Problem**: FileNotFoundError for best_gat_model.pt

**Solution**: Train the model first:

```bash
python 03_train.py
```

### Slow Execution

**Problem**: Analysis taking too long

**Solution**: Reduce scope:

```bash
python counterfactual.py --num_molecules 20 --num_variants 3
```

### Memory Issues

**Problem**: Out of memory errors

**Solution**:

1. Reduce number of molecules
2. Reduce number of variants
3. Use smaller models (adjust in model.py)

### Visualization Errors

**Problem**: PIL/Pillow not installed

**Solution**:

```bash
pip install pillow
```

### Invalid SMILES Errors

**Problem**: "Invalid SMILES" warnings during analysis

**Solution**: This is normal - some generated counterfactuals may be invalid. The script automatically skips these.

---

## File Structure

After running the pipeline:

```
Drug--Induced-Cardiotoxicity--main/
├── counterfactual.py                    ← Main script (RUN THIS)
├── counterfactual_generator.py          ← Generates counterfactuals
├── counterfactual_visualization.py      ← Creates visualizations
├── COUNTERFACTUAL_README.md             ← Full documentation
├── IMPLEMENTATION_SUMMARY.md            ← Technical details
├── validate_counterfactual.py           ← Validation script
│
├── outputs/
│   ├── best_gat_model.pt                ← Trained model
│   ├── counterfactual_report.csv        ← RESULTS (main output)
│   └── counterfactual_summary.json      ← STATISTICS
│
└── counterfactual_examples/
    ├── README.md
    └── *.png                            ← VISUALIZATIONS
```

---

## Success Indicators

After analysis completes, check:

✓ **CSV Report exists**: `outputs/counterfactual_report.csv`

- Should have 100+ rows (50 molecules × 2-5 variants)
- Check: `wc -l outputs/counterfactual_report.csv`

✓ **JSON Summary exists**: `outputs/counterfactual_summary.json`

- Check: `cat outputs/counterfactual_summary.json`

✓ **Visualizations created**: `counterfactual_examples/*.png`

- Check: `ls -la counterfactual_examples/ | wc -l`

✓ **Metrics look reasonable**:

- Mean probability drop: 0.10 - 0.40
- Classification changes: 5% - 50%
- Valid counterfactuals: > 80%

---

## Next Steps

### 1. Understand Results

Read `COUNTERFACTUAL_README.md` for detailed interpretation

### 2. Visualize

Check PNG images in `counterfactual_examples/` folder

### 3. Statistically Analyze

Use Python snippets above to test hypotheses

### 4. Validate Findings

Compare with domain knowledge from hERG literature:

- Known hERG pharmacophores: basic amines, aromatic rings
- Should see large probability drops for these features?

### 5. Publish

Results are ready for reports, papers, regulatory submissions

---

## Common Questions

**Q: Why do some counterfactuals have Δprob ≈ 0?**
A: Those atoms aren't actually important for predictions. The attention mechanism may have picked them up due to correlation, not causation.

**Q: Why is there a "random_removal" modification type?**
A: It's a control baseline. Comparing attention-guided vs random shows whether GAT is actually identifying causal atoms.

**Q: Can I run this on new molecules?**
A: Yes - you'll need to:

1. Convert SMILES to graph using `02_graph_construction.py`
2. Create a simple script using `CounterfactualValidator` class
3. Analyze directly

**Q: How do I cite this work?**
A: See COUNTERFACTUAL_README.md for citation template

---

## Support Resources

| Resource          | Location                  | Use For                 |
| ----------------- | ------------------------- | ----------------------- |
| Usage Guide       | COUNTERFACTUAL_README.md  | How to use everything   |
| Technical Details | IMPLEMENTATION_SUMMARY.md | How it works internally |
| Quick Start       | This file                 | 5-minute overview       |
| Source Code       | Inline docstrings         | Function documentation  |

---

**Ready to go?**

```bash
python counterfactual.py --num_molecules 20
```

Start with 20 molecules (5 minutes), then scale up!
