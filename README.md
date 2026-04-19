# AutoDDPM Implementation

Production-oriented implementation for single-subject autoDDPM inference, including:

- a managed CLI (`./auto`)
- clean inference entrypoint (`inference_clean.py`)
- model/runtime files needed for inference

## What This Repo Focuses On

- Single-subject T1w anomaly detection using pretrained weights
- Slurm-friendly operations (`start`, `stop`, `logs`, `checks`)
- Minimal documentation and operational workflow

## Quick Usage

```bash
cd /path/to/autoDDPM
./auto checks --input "/path/to/subject_T1w.nii.gz" --output "results_subject"
./auto start --input "/path/to/subject_T1w.nii.gz" --output "results_subject" --threshold -1
./auto logs --follow
```

See `QUICK_START.md` for full command details.

## Threshold Guidance

- `0.13`: paper-style fixed threshold
- `-1`: per-scan auto threshold (recommended when no healthy calibration set is available)

## Reference

Method reference paper:
- https://arxiv.org/abs/2305.19643
