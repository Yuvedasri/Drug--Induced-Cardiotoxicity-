"""
Counterfactual molecule visualization module.

Generates side-by-side visualizations of original and counterfactual molecules
with highlighted important atoms and modified atoms.
"""

import os
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class CounterfactualVisualizer:
    """
    Generates and saves visualizations of counterfactual molecules.
    """
    
    DEFAULT_IMG_SIZE = (400, 400)
    
    @staticmethod
    def highlight_atoms(mol, atom_indices, color=(1, 0, 0), atom_size=1.5):
        """
        Create atom highlighting map for RDKit Draw functions.
        
        Args:
            mol: RDKit molecule object
            atom_indices: List of atom indices to highlight
            color: RGB tuple for highlight color (0-1 range)
            atom_size: Size multiplier for highlighted atoms
            
        Returns:
            tuple: (highlight_atoms, highlight_bonds, highlight_atom_colors)
        """
        highlight_atoms = set(atom_indices)
        highlight_bonds = set()
        highlight_atom_colors = {int(idx): color for idx in atom_indices}
        
        return highlight_atoms, highlight_bonds, highlight_atom_colors
    
    @staticmethod
    def generate_2d_coords(mol):
        """
        Generate 2D coordinates for molecule if not present.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            RDKit molecule object with 2D coordinates
        """
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass
        return mol
    
    @classmethod
    def draw_side_by_side(cls, original_smiles, counterfactual_smiles, 
                          important_atoms, modified_atoms, 
                          output_path, title="Counterfactual", 
                          img_size=None):
        """
        Generate side-by-side visualization of original and counterfactual molecules.
        
        Args:
            original_smiles: SMILES of original molecule
            counterfactual_smiles: SMILES of counterfactual molecule
            important_atoms: List of important atom indices in original
            modified_atoms: List of modified atom indices in counterfactual
            output_path: Path to save the output image
            title: Title for the visualization
            img_size: Tuple of (width, height) for each molecule image
            
        Returns:
            bool: True if successful, False otherwise
        """
        if img_size is None:
            img_size = cls.DEFAULT_IMG_SIZE
        
        try:
            # Create molecules
            original_mol = Chem.MolFromSmiles(original_smiles)
            counterfactual_mol = Chem.MolFromSmiles(counterfactual_smiles)
            
            if original_mol is None or counterfactual_mol is None:
                return False
            
            # Generate 2D coordinates
            original_mol = cls.generate_2d_coords(original_mol)
            counterfactual_mol = cls.generate_2d_coords(counterfactual_mol)
            
            # Draw original with highlighted important atoms (red)
            highlight_atoms_orig, _, highlight_colors_orig = cls.highlight_atoms(
                original_mol, important_atoms, color=(1, 0, 0)
            )
            
            img_original = Draw.MolToImage(
                original_mol,
                size=img_size,
                highlightAtoms=highlight_atoms_orig,
                highlightAtomColors=highlight_colors_orig,
                kekulize=True
            )
            
            # Draw counterfactual with highlighted modified atoms (blue)
            highlight_atoms_cf, _, highlight_colors_cf = cls.highlight_atoms(
                counterfactual_mol, modified_atoms, color=(0, 0, 1)
            )
            
            img_counterfactual = Draw.MolToImage(
                counterfactual_mol,
                size=img_size,
                highlightAtoms=highlight_atoms_cf,
                highlightAtomColors=highlight_colors_cf,
                kekulize=True
            )
            
            # Combine images side-by-side
            from PIL import Image, ImageDraw, ImageFont
            
            # Calculate combined image size
            total_width = img_size[0] * 2 + 20  # 20 pixel gap
            total_height = img_size[1] + 60  # Extra space for title
            
            combined_img = Image.new('RGB', (total_width, total_height), color='white')
            
            # Paste images
            combined_img.paste(img_original, (10, 50))
            combined_img.paste(img_counterfactual, (img_size[0] + 10, 50))
            
            # Add title and labels
            draw = ImageDraw.Draw(combined_img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            
            # Title
            draw.text((10, 10), title, fill='black', font=font)
            
            # Labels
            draw.text((10, img_size[1] + 10), "Original (Red = Important)", fill='black', font=font)
            draw.text((img_size[0] + 10, img_size[1] + 10), "Counterfactual (Blue = Modified)", 
                     fill='black', font=font)
            
            # Save image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_img.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return False
    
    @classmethod
    def draw_grid(cls, molecules_data, output_path, grid_rows=2):
        """
        Generate grid visualization of multiple molecules.
        
        molecules_data: List of dicts with keys:
            - 'smiles': molecule SMILES
            - 'label': label text for molecule
            - 'highlight_atoms': list of atom indices to highlight
            - 'color': RGB tuple for highlight color
            
        output_path: Path to save grid image
        grid_rows: Number of rows in grid
        
        Returns:
            bool: True if successful
        """
        if len(molecules_data) == 0:
            return False
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Prepare molecules
            mols = []
            for data in molecules_data:
                mol = Chem.MolFromSmiles(data['smiles'])
                if mol is not None:
                    mol = cls.generate_2d_coords(mol)
                    mols.append(mol)
                else:
                    mols.append(None)
            
            # Draw grid
            grid_cols = (len(mols) + grid_rows - 1) // grid_rows
            img_size = (300, 300)
            
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=grid_cols,
                subImgSize=img_size,
                legends=[data.get('label', '') for data in molecules_data],
                returnPNG=False
            )
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error generating grid visualization: {str(e)}")
            return False
    
    @classmethod
    def draw_with_importance_score(cls, smiles, importance_scores, output_path,
                                   title="Molecular Attention Map"):
        """
        Create a single molecule image with atom importance indicated by color intensity.
        
        Args:
            smiles: Molecule SMILES string
            importance_scores: Dict mapping atom index -> importance score (0-1)
            output_path: Path to save image
            title: Title for visualization
            
        Returns:
            bool: True if successful
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            mol = cls.generate_2d_coords(mol)
            
            # Create color map based on importance scores
            highlight_atom_colors = {}
            for atom_idx, score in importance_scores.items():
                # Color scale from white (low) to red (high)
                # score ranges 0-1
                r = score
                g = 1 - score
                b = 1 - score
                highlight_atom_colors[int(atom_idx)] = (r, g, b)
            
            highlight_atoms = set(importance_scores.keys())
            
            img = Draw.MolToImage(
                mol,
                size=(400, 400),
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_atom_colors,
                kekulize=True
            )
            
            # Add title
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            
            draw.text((10, 10), title, fill='black', font=font)
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error generating importance score visualization: {str(e)}")
            return False
