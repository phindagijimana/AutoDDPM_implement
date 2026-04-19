# autoDDPM Quick Start

This project supports two usage modes:

- `inference_clean.py`: single-subject inference with a pretrained weight (no dataset CSV needed)
- `core/Main.py`: official framework evaluation/training (requires dataset CSVs and data layout)

## Single-Subject Inference (Recommended)

1. Activate environment:

```bash
source ~/Documents/autoddpm_env/bin/activate
cd /path/to/autoDDPM
```

2. Run inference:

```bash
python inference_clean.py \
  --input "/path/to/subject_T1w.nii.gz" \
  --output "results_subject/" \
  --model_path "./latest_model.pt" \
  --noise_recon 200 \
  --noise_inpaint 50 \
  --resample_steps 5 \
  --threshold -1 \
  --batch_size 8 \
  --device cuda
```

Notes:
- Use `--threshold 0.13` to follow the paper's fixed threshold setting.
- Use `--threshold -1` for per-scan auto-thresholding when no healthy calibration set is available.

## Production CLI (`./auto`)

Use the built-in CLI for install/start/stop/logs/checks.

```bash
cd /path/to/autoDDPM

# environment/dependency setup
./auto install --venv ~/Documents/autoddpm_env --with-wandb

# preflight checks
./auto checks --input "/path/to/subject_T1w.nii.gz" --output "results_subject"

# start inference on Slurm (default)
./auto start \
  --input "/path/to/subject_T1w.nii.gz" \
  --output "results_subject" \
  --threshold -1

# follow logs for latest run
./auto logs --follow

# stop latest run
./auto stop
```

## Expected Outputs

The output folder contains:

- `*_anomaly_map.nii.gz` (main output)
- `*_reconstruction.nii.gz`
- `*_binary_85th.nii.gz`
- `*_binary_90th.nii.gz`
- `*_binary_95th.nii.gz`
- `*_middle_slice.png`
- `*_stats.txt`

## Full Framework Pipeline (Optional)

Run only if you need official dataset-level evaluation:

```bash
python core/Main.py --config_path ./projects/autoddpm/autoddpm.yaml
```

This mode requires valid dataset CSV files and corresponding image paths.
