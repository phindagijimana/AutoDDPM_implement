#!/usr/bin/env python
"""
Test autoDDPM Setup
===================

Quick verification script to check if autoDDPM is properly set up.

Usage:
    python test_setup.py
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed."""
    print("Testing package imports...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'nibabel': 'nibabel',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'lpips': 'lpips',
        'cv2': 'opencv-python',
        'yaml': 'PyYAML',
        'wandb': 'wandb',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All required packages installed!")
        return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    import torch
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("  ⚠️  CUDA not available (will use CPU)")
        return False


def test_model():
    """Test if model file exists and can be loaded."""
    print("\nTesting model...")
    import torch
    
    model_path = './latest_model.pt'
    if not os.path.exists(model_path):
        print(f"  ✗ Model not found at {model_path}")
        print(f"    Download from: https://www.dropbox.com/s/ooq7vdp9fp4ufag/latest_model.pt.zip?dl=0")
        return False
    
    print(f"  ✓ Model file exists: {model_path}")
    
    # Try to load model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"  ✓ Model loads successfully")
        
        if 'model_weights' in checkpoint:
            print(f"  ✓ Model weights found in checkpoint")
        if 'epoch' in checkpoint:
            print(f"  ✓ Trained for {checkpoint['epoch']} epochs")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False


def test_model_creation():
    """Test if DDPM model can be instantiated."""
    print("\nTesting DDPM model creation...")
    
    try:
        from model_zoo.ddpm import DDPM
        
        model = DDPM(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=[128, 256, 256],
            attention_levels=[False, True, True],
            num_res_blocks=1,
            num_head_channels=256,
        )
        
        print(f"  ✓ DDPM model created successfully")
        
        # Test forward pass with dummy input
        import torch
        dummy_input = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            # Just test that forward pass doesn't crash
            timesteps = torch.randint(0, 1000, (1,))
            noise = torch.randn_like(dummy_input)
            output = model(dummy_input, noise=noise, timesteps=timesteps)
        
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_data():
    """Test if sample data exists."""
    print("\nTesting sample data...")
    
    sample_dir = './TRACK_Sample_Image/'
    if not os.path.exists(sample_dir):
        print(f"  ⚠️  Sample data directory not found: {sample_dir}")
        return False
    
    samples = [f for f in os.listdir(sample_dir) if f.endswith(('.nii', '.nii.gz'))]
    
    if not samples:
        print(f"  ⚠️  No NIfTI files found in {sample_dir}")
        return False
    
    print(f"  ✓ Sample data directory exists: {sample_dir}")
    print(f"  ✓ Found {len(samples)} sample files:")
    for sample in samples:
        print(f"    - {sample}")
    
    return True


def test_config():
    """Test if config file exists and is valid."""
    print("\nTesting configuration...")
    
    config_path = './projects/autoddpm/autoddpm.yaml'
    if not os.path.exists(config_path):
        print(f"  ✗ Config not found: {config_path}")
        return False
    
    print(f"  ✓ Config exists: {config_path}")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        print(f"  ✓ Config is valid YAML")
        
        # Check key sections
        if 'model' in config:
            print(f"  ✓ Model config found")
        if 'experiment' in config:
            print(f"  ✓ Experiment config found")
        
        return True
    except Exception as e:
        print(f"  ✗ Error parsing config: {e}")
        return False


def print_summary(results):
    """Print summary of test results."""
    print("\n" + "="*60)
    print("SETUP VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("="*60)
    
    if all_passed:
        print("✓ All tests passed! autoDDPM is ready to use.")
        print("\nNext steps:")
        print("  1. Read QUICK_START.md")
        print("  2. Run inference: python inference_clean.py --help")
        print("  3. Test on sample data:")
        print("     python inference_clean.py \\")
        print("       --input TRACK_Sample_Image/sample_T1w_2.nii.gz \\")
        print("       --output test_results/")
    else:
        print("⚠️  Some tests failed. Please resolve issues before proceeding.")
        print("\nTroubleshooting:")
        print("  - Missing packages? Run: pip install -r pip_requirements.txt")
        print("  - Model missing? Download from Dropbox (see README.md)")
        print("  - Check IMPLEMENTATION_GUIDE.md for detailed setup")
    
    print("="*60 + "\n")
    
    return all_passed


def main():
    print("\n" + "="*60)
    print("autoDDPM Setup Verification")
    print("="*60 + "\n")
    
    results = {}
    
    results['Package Imports'] = test_imports()
    results['CUDA'] = test_cuda()
    results['Model File'] = test_model()
    results['Model Creation'] = test_model_creation()
    results['Sample Data'] = test_sample_data()
    results['Configuration'] = test_config()
    
    success = print_summary(results)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()






