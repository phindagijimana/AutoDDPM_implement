#!/bin/bash
#SBATCH --job-name=autoDDPM_track
#SBATCH --output=logs/autoDDPM_track_%j.out
#SBATCH --error=logs/autoDDPM_track_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:l40s.12g:1
#SBATCH --partition=general

# autoDDPM Batch Job for TRACK Sample Image
# ==========================================
# This script runs autoDDPM inference on TRACK brain MRI samples
# with GPU acceleration for fast processing.

echo "=========================================="
echo "autoDDPM TRACK Sample Processing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Setup: repo root = this script's directory (no hardcoded paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# Activate environment (override with AUTO_DDPM_VENV=/path/to/venv)
echo "Activating environment..."
if [[ -n "${AUTO_DDPM_VENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${AUTO_DDPM_VENV}/bin/activate"
elif [[ -f "${HOME}/.venvs/autoddpm/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.venvs/autoddpm/bin/activate"
elif [[ -f "${HOME}/Documents/autoddpm_env/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/Documents/autoddpm_env/bin/activate"
else
  echo "ERROR: No venv found. Set AUTO_DDPM_VENV to your venv root or install per QUICK_START.md." >&2
  exit 1
fi

# Check GPU
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Optional: TRACK_INPUT=relative/or/absolute/path.nii sbatch run_track_sample.sh
TRACK_INPUT="${TRACK_INPUT:-TRACK_Sample_Image/sample_T1w_2.nii.gz}"

echo "Removing old cache for this input (if any)..."
STEM="${TRACK_INPUT##*/}"
STEM="${STEM%.nii.gz}"
STEM="${STEM%.nii}"
rm -f "autoDDPM_outputs/results_masked_resampled_${STEM}.npz"
rm -f autoDDPM_outputs/results_masked_resampled.npz
echo ""

# Run autoDDPM on single slice (fast test)
echo "=========================================="
echo "Running autoDDPM on TRACK sample (single slice)"
echo "=========================================="
echo "Input: $TRACK_INPUT"
echo "Output: autoDDPM_outputs/"
echo ""

time python scripts_legacy/main.py --input "$TRACK_INPUT" --force

echo ""
echo "=========================================="
echo "Single slice processing complete!"
echo "=========================================="
echo ""

# List outputs
echo "Generated outputs:"
ls -lh autoDDPM_outputs/

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

