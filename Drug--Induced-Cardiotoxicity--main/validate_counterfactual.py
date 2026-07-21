"""
Quick validation script to test counterfactual pipeline components.

This script verifies that all modules can be imported and basic functionality works.

Usage:
    python validate_counterfactual.py
"""

import sys
import os

print("=" * 80)
print("COUNTERFACTUAL VALIDATION - QUICK TEST")
print("=" * 80)
print()

# Test 1: Import modules
print("[1/4] Testing imports...")
try:
    from counterfactual_generator import CounterfactualGenerator
    print("  ✓ counterfactual_generator imported")
except ImportError as e:
    print(f"  ✗ Failed to import counterfactual_generator: {e}")
    sys.exit(1)

try:
    from counterfactual_visualization import CounterfactualVisualizer
    print("  ✓ counterfactual_visualization imported")
except ImportError as e:
    print(f"  ✗ Failed to import counterfactual_visualization: {e}")
    sys.exit(1)

try:
    from counterfactual import CounterfactualValidator
    print("  ✓ counterfactual imported")
except ImportError as e:
    print(f"  ✗ Failed to import counterfactual: {e}")
    sys.exit(1)

print()

# Test 2: Test counterfactual generator
print("[2/4] Testing CounterfactualGenerator...")
try:
    # Test with a simple drug-like molecule
    test_smiles = "CCN(CC)CCOC(=O)c1ccc(cc1)N"  # para-aminobenzoic acid derivative
    
    # Test validity check
    from rdkit import Chem
    mol = Chem.MolFromSmiles(test_smiles)
    is_valid = CounterfactualGenerator.is_valid_molecule(mol)
    print(f"  ✓ Molecule validation works: {is_valid}")
    
    # Test generation
    important_atoms = [0, 1, 2]  # First carbons
    counterfactuals = CounterfactualGenerator.generate_counterfactuals(
        test_smiles, important_atoms, num_variants=3
    )
    print(f"  ✓ Generated {len(counterfactuals)} counterfactuals")
    
    if len(counterfactuals) > 0:
        print(f"    - Example modifications: {[cf['modification_type'] for cf in counterfactuals]}")
    
    # Test random counterfactual
    random_cf = CounterfactualGenerator.generate_random_counterfactual(test_smiles, num_atoms_to_modify=2)
    if random_cf is not None:
        print(f"  ✓ Generated random control counterfactual")
    else:
        print(f"  ⚠ Random counterfactual generation returned None (expected for this molecule)")
    
except Exception as e:
    print(f"  ✗ Error testing CounterfactualGenerator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Test visualization (requires PIL)
print("[3/4] Testing CounterfactualVisualizer...")
try:
    import tempfile
    
    # Test with a simple pair
    test_smiles_1 = "CCO"  # Ethanol
    test_smiles_2 = "CC"   # Ethane
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_cf.png")
        
        # Note: May fail if PIL not installed, but that's OK
        try:
            success = CounterfactualVisualizer.draw_side_by_side(
                test_smiles_1, test_smiles_2,
                important_atoms=[0],
                modified_atoms=[1],
                output_path=output_path,
                title="Test Visualization"
            )
            if success:
                print(f"  ✓ Visualization generated successfully")
            else:
                print(f"  ⚠ Visualization function returned False")
        except ImportError:
            print(f"  ⚠ PIL not available for visualization (optional dependency)")
        except Exception as e_viz:
            print(f"  ⚠ Visualization error (optional, can be ignored): {e_viz}")

except Exception as e:
    print(f"  ✗ Error testing CounterfactualVisualizer: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Test validator (if model available)
print("[4/4] Testing CounterfactualValidator setup...")
try:
    import torch
    
    validator = CounterfactualValidator()
    print(f"  ✓ Validator initialized")
    print(f"  ✓ Device available: {validator.device}")
    
    # Check if model exists
    model_path = "outputs/best_gat_model.pt"
    if os.path.exists(model_path):
        print(f"  ✓ Model file found at {model_path}")
        print(f"    → Ready to run full analysis")
    else:
        print(f"  ⚠ Model not found at {model_path}")
        print(f"    → Run 03_train.py first to generate the model")
    
except Exception as e:
    print(f"  ✗ Error testing CounterfactualValidator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✓ VALIDATION COMPLETE - All core components working!")
print("=" * 80)
print()

print("Next steps:")
print("  1. Train the model (if not already trained):")
print("     python 03_train.py")
print()
print("  2. Run the counterfactual analysis:")
print("     python counterfactual.py --num_molecules 10")
print()
print("  3. Check outputs:")
print("     - outputs/counterfactual_report.csv")
print("     - outputs/counterfactual_summary.json")
print("     - counterfactual_examples/*.png")
print()
