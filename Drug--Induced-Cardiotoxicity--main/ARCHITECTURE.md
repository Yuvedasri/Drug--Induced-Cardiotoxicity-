# Architecture & Component Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COUNTERFACTUAL VALIDATION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

                    INPUT LAYER
                    ───────────
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│Molecular │        │Attention │        │ Trained  │
│ Graphs   │        │Weights   │        │   GAT    │
│ (*.pt)   │        │ (Layer 1-3)       │ Model    │
└──────────┘        └──────────┘        └──────────┘
    │                   │                   │
    └───────────────────┼───────────────────┘
                        │
                        ▼
            ┌──────────────────────┐
            │ CounterfactualValidator
            │  - load_model()      │
            │  - predict()         │
            │  - extract_important │
            │    _atoms()          │
            └──────┬───────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
 ┌──────────────────┐  ┌─────────────────┐
 │ Important Atoms  │  │ Counterfactual  │
 │ (indices)        │  │ SMILES List     │
 │                  │  │                 │
 │ Top-K Selection  │  │ Generation      │
 │ from Attention   │  │ Strategies      │
 └────────┬─────────┘  └────────┬────────┘
          │                     │
          └─────────────┬───────┘
                        │
                        ▼
         ┌──────────────────────────┐
         │CounterfactualGenerator   │
         │  Modification Strategies │
         │                          │
         │  • atom_removal          │
         │  • substituent_reduction │
         │  • benzene→cyclohexane   │
         │  • pyridine→piperidine   │
         │  • halogen→hydrogen      │
         │  • nitro→hydrogen        │
         │  • random (control)      │
         └──────────┬───────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │ RDKit Validation         │
         │                          │
         │ ✓ Chemical Validity      │
         │ ✓ Molecular Weight       │
         │ ✓ Sanitization           │
         └──────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    VALID▼                     ▼INVALID
  ┌─────────────┐          (Skip)
  │ Counterfactual
  │  SMILES     │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────┐
  │ Convert to Graph    │
  │ (mol_to_graph)      │
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │ Predict Toxicity    │
  │ P(counterfactual)   │
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │ Calculate Metrics   │
  │                     │
  │ • Δprob             │
  │ • Class change?     │
  │ • Validity          │
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │ Per-Molecule Result │
  │                     │
  │ • Original prob     │
  │ • Important atoms   │
  │ • Counterfactuals   │
  │ • Statistics        │
  └──────┬──────────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
    ┌──────────┐    ┌──────────────────┐
    │ Report   │    │ Visualization    │
    │ CSV      │    │                  │
    │ JSON     │    │ Side-by-side     │
    │ Summary  │    │ PNG images       │
    └──────────┘    └──────────────────┘
```

## Module Dependencies

```
counterfactual.py (MAIN ORCHESTRATOR)
│
├─→ model.py
│   (Load trained GATv2 network)
│
├─→ 02_graph_construction.py
│   (Convert SMILES → molecular graphs)
│
├─→ counterfactual_generator.py
│   (Generate valid counterfactuals)
│   │
│   └─→ rdkit (RDKit cheminformatics)
│
├─→ counterfactual_visualization.py
│   (Create PNG visualizations)
│   │
│   ├─→ rdkit.Chem.Draw
│   └─→ PIL/Pillow
│
└─→ Standard libraries
    (torch, numpy, pandas, json, etc.)
```

## Key Components

### 1. CounterfactualValidator Class

```
┌────────────────────────────────────────┐
│  CounterfactualValidator               │
├────────────────────────────────────────┤
│ Attributes:                            │
│  • model: trained GATv2                │
│  • device: torch.device                │
│  • model_path: checkpoint location     │
├────────────────────────────────────────┤
│ Methods:                               │
│  • load_model(node_dim, edge_dim)      │
│  • get_node_attention(data)            │
│  • predict(data) → probability         │
│  • extract_important_atoms(data, k)    │
│  • analyze_molecule(data, params)      │
│  • visualize_result(result)            │
└────────────────────────────────────────┘
```

### 2. CounterfactualGenerator Class

```
┌────────────────────────────────────────┐
│  CounterfactualGenerator               │
├────────────────────────────────────────┤
│ Static Methods:                        │
│  • is_valid_molecule(mol)              │
│  • molecular_weight_is_reasonable()    │
│  • generate_counterfactuals()          │
│  • generate_random_counterfactual()    │
├────────────────────────────────────────┤
│ Transformation Strategies:             │
│  • REPLACEMENT_TRANSFORMATIONS (list) │
│    - aromatic rings                    │
│    - amines                            │
│    - halogens                          │
│    - nitro groups                      │
│    - carbonyls                         │
└────────────────────────────────────────┘
```

### 3. CounterfactualVisualizer Class

```
┌────────────────────────────────────────┐
│  CounterfactualVisualizer              │
├────────────────────────────────────────┤
│ Static Methods:                        │
│  • highlight_atoms(mol, indices)       │
│  • generate_2d_coords(mol)             │
│  • draw_side_by_side(orig, cf)         │
│  • draw_grid(mol_list)                 │
│  • draw_with_importance_score()        │
└────────────────────────────────────────┘
```

## Data Flow: Single Molecule Analysis

```
Input: PyTorch Geometric Data object for one molecule
────────────────────────────────────────────────────

STEP 1: PREDICTION
  data.x (node features)
  data.edge_index (connectivity)
  data.edge_attr (bond features)
         ↓
    [Forward pass through GATv2]
         ↓
    original_probability ∈ [0, 1]
    attn_weights (per layer)

STEP 2: IMPORTANCE EXTRACTION
  attn_weights
         ↓
    [Aggregate across layers]
         ↓
    node importance scores
         ↓
    [Sort by score]
         ↓
    important_atom_indices = Top-K

STEP 3: COUNTERFACTUAL GENERATION
  data.smiles + important_atom_indices
         ↓
    [Apply transformation strategy 1]
         ↓
    candidate_smiles_1
         ↓
    [Validate with RDKit]
         ↓
    ✓ VALID → counterfactual_1.smiles
    ✗ INVALID → Skip

    [Repeat for strategies 2, 3, 4, 5]
         ↓
    counterfactual_smiles_list = [cfsmiles_1, cfsmiles_2, ...]

STEP 4: PREDICTION ON COUNTERFACTUALS
  For each cfsmiles:
    [convert to molecular graph]
         ↓
    [predict toxicity]
         ↓
    cf_probability, cf_class
         ↓
    prob_drop = original_prob - cf_prob
    class_changed = (orig_class != cf_class)

STEP 5: RESULTS COMPILATION
  result = {
    'smiles': original SMILES,
    'original_probability': float,
    'important_atom_indices': [int],
    'counterfactuals': [
      {
        'smiles': cfsmiles,
        'modification_type': str,
        'prob_drop': float,
        'class_changed': bool,
        ...
      },
      ...
    ],
    'mean_prob_drop': float,
    'n_class_changes': int,
    ...
  }

STEP 6: VISUALIZATION
  For top 3 counterfactuals:
    [Call draw_side_by_side]
         ↓
    PNG image saved

Output: Result dict + PNG files
──────────────────────────────
```

## Computational Complexity

For N molecules with K important atoms and V counterfactual variants:

| Operation                 | Complexity            | Notes                      |
| ------------------------- | --------------------- | -------------------------- |
| Attention extraction      | O(N × L × E)          | L=layers, E=edges          |
| Importance selection      | O(N × K log K)        | Sorting K atoms            |
| Counterfactual generation | O(N × V × G)          | G=graph validation         |
| Counterfactual prediction | O(N × V × L × N²)     | N=atoms per molecule       |
| Visualization             | O(N × V × pixels)     | PNG generation             |
| **Total**                 | **O(N × V × L × N²)** | Dominated by recomputation |

**Typical runtime**:

- Per molecule: 5-15 seconds (V=5 variants)
- 50 molecules: 5-12 minutes
- 200 molecules: 20-50 minutes
- GPU acceleration: 2-3x speedup (if CUDA available)

## Memory Usage

Per molecule analysis:

- Model weights: ~50 MB (shared)
- Molecule data: ~1-5 MB
- Counterfactuals: ~5-20 MB temporary
- **Total per analysis**: ~100-200 MB RAM

Peak usage during batch: Multiplied by number of parallel processes

## File I/O Operations

```
READ:
├─ Input
│  ├─ best_gat_model.pt (model weights)
│  └─ molecular_graphs.pt (test graphs)
│
└─ Processing
   └─ 02_graph_construction.py (import)

WRITE:
├─ outputs/
│  ├─ counterfactual_report.csv (growing)
│  └─ counterfactual_summary.json (at end)
│
└─ counterfactual_examples/
   └─ *.png (one per counterfactual variant)

Total output: ~50-500 MB for 50-200 molecules
```

## Error Handling Deep Dive

```
Molecule Analysis
├─ Invalid SMILES
│  └─ Skip molecule
│
├─ Graph conversion fails
│  └─ Return error, skip
│
├─ Model prediction error
│  └─ Catch exception, retry
│
├─ Counterfactual generation
│  ├─ Ring chemistry error → Skip variant
│  ├─ RDKit validation fails → Skip variant
│  ├─ MW out of range → Skip variant
│  └─ Duplicate SMILES → Skip variant
│
├─ Graph conversion of counterfactual
│  └─ Skip counterfactual
│
├─ Visualization generation
│  └─ Log warning, continue
│
└─ Report writing
   └─ Catch IO errors
```

## Optimization Opportunities

### Quick Wins (Easy to implement)

- [ ] Cache 2D coordinates for molecules
- [ ] Batch predictions for multiple counterfactuals
- [ ] Parallel processing with multiprocessing
- [ ] Skip full graph reconstruction for similar molecules

### Medium Effort

- [ ] GPU acceleration (CUDA for PyTorch)
- [ ] Cython compilation for hot loops
- [ ] SMARTS pattern compilation caching

### Hard/Advanced

- [ ] Reactive graph modification (in-place)
- [ ] Custom RDKit fork optimizations
- [ ] Distributed processing (Ray, Dask)
- [ ] Approximate importance with sampling

## Integration Points

### With Existing Pipeline

```
03_train.py
    ↓ produces
outputs/best_gat_model.pt ← counterfactual.py reads
outputs/molecular_graphs.pt ← counterfactual.py reads

04_causal_analysis.py (masking baseline)
    vs.
counterfactual.py (this implementation)
    → Complementary approaches!
```

### With Downstream Applications

```
counterfactual_report.csv
    ↓
Statistical analysis (Python/R)
    ↓
Regulatory submissions
Patent applications
Publication in journals
```

---

## Glossary

| Term                      | Definition                                           |
| ------------------------- | ---------------------------------------------------- |
| **Attention weights**     | Per-head importance scores from GATv2 layers         |
| **Node importance**       | Aggregated attention scores per atom                 |
| **Top-K atoms**           | Most important atoms (by attention)                  |
| **Counterfactual**        | Modified version of molecule (SMILES)                |
| **Modification strategy** | Rule for replacing atoms (e.g., benzene→cyclohexane) |
| **Probability drop**      | Δprob = P(original) - P(counterfactual)              |
| **Class change**          | Did prediction flip from toxic↔non-toxic?            |
| **Validity**              | Is counterfactual chemically sound?                  |
| **SMARTS**                | Simplified Molecular Input Line Entry System         |

---

This architecture is designed for clarity, modularity, and extensibility.
Each component can be tested, modified, or replaced independently!
