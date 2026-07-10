"""
Example: Batch prediction using 06_predict_new_drug programmatically.
Demonstrates how to import and use the predict() function for multiple molecules.
"""
from importlib import import_module

# Import the predict function
predict_mod = import_module("06_predict_new_drug")
predict = predict_mod.predict

# Example drug candidates to test
test_molecules = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Terfenadine", "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c2ccccc2"),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
]

print("\n" + "="*70)
print("BATCH CARDIOTOXICITY PREDICTION")
print("="*70)

results = []
for name, smiles in test_molecules:
    print(f"\nPredicting: {name}")
    result = predict(smiles)
    results.append((name, result))
    
    if result['valid'] and not result['error']:
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Classification: {result['classification']}")
        print(f"  Causal necessity gap: {result['causal_drop'] - result['random_drop']:.4f}")
    else:
        print(f"  ERROR: {result['error']}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
toxic_count = sum(1 for _, r in results if r['valid'] and r['probability'] >= 0.5)
print(f"Total molecules tested: {len(results)}")
print(f"Predicted as TOXIC: {toxic_count}")
print(f"Predicted as NON-TOXIC: {len(results) - toxic_count}")
print("="*70 + "\n")
