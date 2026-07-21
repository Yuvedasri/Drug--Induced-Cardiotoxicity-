# Data Setup Instructions

## Required Dataset

To run this pipeline, you need to obtain the hERG Karim dataset:

### Primary Dataset (Required)
**File:** `hERG_Karim_Dataset.xlsx`  
**Location:** Place in `data/` folder  
**Description:** Balanced dataset of 13,445 molecules with hERG activity labels

### Optional External Validation Dataset
**File:** `tox21-herg-u2os-p1.aggregrated.txt`  
**Location:** Place in `data/` folder  
**Description:** Tox21 hERG dataset for external validation (script 05)

## Why These Files Are Not Included

Due to GitHub's file size limitations and data licensing considerations:
- Large generated files (trained models, molecular graphs) are excluded
- Dataset files should be obtained from official sources
- The `.gitignore` prevents accidental upload of data files

## After Adding Data Files

Once you've placed the dataset in the `data/` folder, run the pipeline:

```bash
python 01_data_prep.py
python 02_graph_construction.py
python 03_train.py
python 04_causal_analysis.py
python 05_external_validation.py  # optional, requires Tox21 data
python 06_predict_new_drug.py "YOUR_SMILES_HERE"
```

## Generated Files

The pipeline will create files in `outputs/`:
- `clean_herg_dataset.csv` (~1.5 MB)
- `molecular_graphs.pt` (~119 MB)
- `best_gat_model.pt` (~1.3 MB)
- `test_metrics.json`, `causal_summary.json`
- Prediction visualizations (PNG files)

These generated files are excluded from the repository but will be created when you run the scripts.
