#!/bin/bash
# Setup Script for autoDDPM
# =========================

echo "============================================================"
echo "autoDDPM Environment Setup"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if conda is available
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓${NC} Conda is available"
    
    # Ask if user wants to create a new environment
    read -p "Create new conda environment 'autoddpm'? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating conda environment..."
        conda create -n autoddpm python=3.8 -y
        echo -e "${GREEN}✓${NC} Environment created"
        
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate autoddpm"
        echo ""
        echo "Then install packages with:"
        echo "  pip install -r pip_requirements.txt"
        echo "  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠${NC}  Conda not found"
    echo "You can install packages in your current Python environment"
fi

echo ""
echo "============================================================"
echo "Installation Steps"
echo "============================================================"
echo ""

echo "1. Install PyTorch (choose based on your CUDA version):"
echo ""
echo "   With CUDA 11.1:"
echo "   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html"
echo ""
echo "   With CUDA 10.2:"
echo "   pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html"
echo ""
echo "   CPU only:"
echo "   pip install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html"
echo ""

echo "2. Install other requirements:"
echo "   pip install -r pip_requirements.txt"
echo ""

echo "3. Download pre-trained model (if not already downloaded):"
echo "   wget https://www.dropbox.com/s/ooq7vdp9fp4ufag/latest_model.pt.zip?dl=0 -O latest_model.pt.zip"
echo "   unzip latest_model.pt.zip"
echo ""

echo "4. Test setup:"
echo "   python test_setup.py"
echo ""

echo "5. Run inference:"
echo "   python inference_clean.py --input TRACK_Sample_Image/sample_T1w_2.nii.gz --output results/"
echo ""

echo "============================================================"
echo "Quick Reference"
echo "============================================================"
echo ""
echo "Documentation:"
echo "  - QUICK_START.md          Fast-track guide"
echo "  - IMPLEMENTATION_GUIDE.md Complete guide"
echo "  - PROJECT_SUMMARY.md      Project organization"
echo ""
echo "Scripts:"
echo "  - inference_clean.py      Main inference script (RECOMMENDED)"
echo "  - test_setup.py           Verify installation"
echo "  - scripts_legacy/         Your original scripts (preserved)"
echo ""
echo "============================================================"






