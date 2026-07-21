# Counterfactual Validation Examples

This directory contains visualizations and results from the counterfactual validation analysis.

## Contents

- `*.png`: Side-by-side visual comparisons of original and counterfactual molecules
  - **Left (RED highlights)**: Original molecule with important atoms highlighted
  - **Right (BLUE highlights)**: Counterfactual molecule with modified atoms highlighted

## File Naming Convention

Filenames follow the pattern: `[molecule_smiles]_[toxicity_probability]_cf[index].png`

- `molecule_smiles`: Sanitized SMILES representation (first 50 chars)
- `toxicity_probability`: Original predicted toxicity probability (0-100 scale)
- `cf[index]`: Counterfactual variant index (cf0, cf1, cf2...)

## Output Metrics

See `../outputs/counterfactual_report.csv` for detailed metrics:

- Original SMILES
- Counterfactual SMILES
- Number of atoms modified
- Types of modifications applied
- Probability drops
- Classification changes
- Molecular validity

## Summary Statistics

See `../outputs/counterfactual_summary.json` for aggregate statistics:

- Mean/median/max probability drops
- Percentage of molecules with prediction changes
- Parameter settings used for analysis
