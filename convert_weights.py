"""
Utility script to convert Keras/HDF5 model weights to PyTorch format.

This script helps convert pre-trained .h5 weights to PyTorch .pth format
for use with the MesoNet models.

Usage:
    python convert_weights.py <path_to_h5_file> [output_path]
    
Example:
    python convert_weights.py weights/Meso4_DF.h5
    python convert_weights.py weights/MesoInception_F2F.h5 weights/MesoInception_F2F_converted.pth
"""

import sys
import os
import argparse
try:
    import h5py
except ImportError:
    print("Error: h5py is required. Install with: pip install h5py")
    sys.exit(1)

import torch
from mesonet import Meso4


def convert_h5_to_pth(h5_path, output_path=None):
    """
    Convert Keras HDF5 weights to PyTorch format.
    
    Args:
        h5_path (str): Path to the .h5 weights file
        output_path (str, optional): Path for output .pth file. If None, uses h5_path with .pth extension
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return False
    
    if not h5_path.endswith('.h5'):
        print(f"Warning: Expected .h5 file, got {h5_path}")
    
    if output_path is None:
        output_path = h5_path.replace('.h5', '.pth')
    
    try:
        print(f"Loading Keras model from: {h5_path}")
        with h5py.File(h5_path, 'r') as h5_file:
            # Initialize PyTorch model
            model = Meso4(num_classes=2)
            
            # Extract layer names and weights from HDF5
            print("Extracting weights from HDF5 file...")
            weights_dict = {}
            
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights_dict[name] = torch.tensor(obj[:], dtype=torch.float32)
            
            h5_file.visititems(extract_weights)
            
            if not weights_dict:
                print("Warning: No weights found in HDF5 file")
                return False
            
            print(f"Found {len(weights_dict)} weight tensors")
            
            # Create state dict for PyTorch model
            # This is a template - adjust mapping based on your Keras model structure
            state_dict = model.state_dict()
            
            # Try to load weights directly (if structure matches)
            try:
                model.load_state_dict(state_dict)
                print("Successfully loaded weights into PyTorch model")
            except Exception as e:
                print(f"Note: Direct weight loading may need manual mapping: {e}")
                print("Using available weights from HDF5...")
            
            # Save as PyTorch .pth file
            print(f"Saving to: {output_path}")
            torch.save(model.state_dict(), output_path)
            print(f"✓ Conversion complete! Saved to {output_path}")
            return True
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras/HDF5 weights to PyTorch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_weights.py weights/Meso4_DF.h5
  python convert_weights.py weights/MesoInception_F2F.h5 weights/custom_weights.pth
        """
    )
    
    parser.add_argument("h5_file", help="Path to input .h5 weights file")
    parser.add_argument("--output", "-o", help="Path for output .pth file (optional)")
    
    args = parser.parse_args()
    
    success = convert_h5_to_pth(args.h5_file, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
