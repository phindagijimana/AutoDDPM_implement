# Builder Review — autoDDPM (Bercea et al. 2023) + production inference implementation

Concise review using the same builder-oriented dimensions reflected in `LeGUI_br.md` (usability, reproducibility, performance, generalization, clinical relevance, interpretability, integration, and limitations), adapted for this repository's single-subject production workflow.

**Primary references**
- Paper: https://arxiv.org/abs/2305.19643
- Upstream code: https://github.com/ci-ber/autoDDPM

---

## Context

autoDDPM is a diffusion-based anomaly detection method for medical imaging centered on:

1. **Mask**: initial anomaly likelihood map from reconstruction and perceptual/residual differences  
2. **Stitch**: combine original context with pseudo-healthy regions  
3. **Re-sample**: inpaint for coherent pseudo-healthy reconstruction

This implementation focuses on **single-subject T1w inference** with pretrained weights and operational controls via `./auto` (`install`, `start`, `stop`, `logs`, `checks`).

---

## Platform fit and reproducibility

### Usability
- Strong for deployment: one command surface for operational runs and log management.
- Practical defaults for production-style use (`noise_recon=200`, `noise_inpaint=50`, `resample_steps=5`, threshold configurable).
- Main operational friction remains GPU/runtime setup and model weight provisioning.

### Reproducibility
- Good infrastructure reproducibility through pinned command paths and explicit parameterized CLI.
- Scientific reproducibility is method-consistent, but threshold behavior remains **data-distribution dependent**.
- For sites without healthy calibration sets, `threshold=-1` provides a documented fallback (auto per-scan thresholding).

---

## Performance, generalization, and comparison

### Performance
- Inference is tractable for full volumes on GPU (minutes-scale), with clear speed/memory tradeoffs via `batch_size`.
- Runtime/throughput are sensitive to hardware class and scheduler environment.

### Generalization
- Method generalizes as anomaly scoring, but absolute binary threshold transfer is limited across domains/scanners.
- Fixed `0.13` is paper-aligned for the original setup; local adaptation is usually required for robust operating points.

### Comparison signal
- Compared with pure reconstruction-difference baselines, autoDDPM's stitch + re-sample pipeline improves structural coherence of pseudo-healthy reconstructions and can improve anomaly localization stability.

---

## Clinical relevance, interpretability, and integration

### Clinical relevance
- Useful as a **research/decision-support anomaly localization tool** for prioritization and QC workflows.
- Not a standalone diagnostic system; outputs require expert interpretation and contextual imaging review.

### Interpretability
- Strength: continuous anomaly maps + binary masks + reconstruction artifacts are inspectable and auditable.
- Limitation: final binary interpretation remains threshold-sensitive and site-dependent.

### Integration potential
- Good fit for HPC/Slurm operations through the repository CLI and log controls.
- Can be inserted into existing MRI processing pipelines as an inference stage without requiring full dataset-level framework runs.

---

## Limitations and failure modes

- Threshold calibration remains the dominant source of site-specific variability.
- False positives can increase in scans with artifacts or domain shift.
- Full framework evaluation (`core/Main.py`) still requires dataset CSV structure and curated assets; this repo intentionally minimizes to inference essentials.

---

## Builder verdict

This implementation is **production-practical for single-subject autoDDPM inference**: it preserves the method's core logic, adds operational reliability, and reduces setup complexity.  
The main residual risk is **threshold governance** across domains; teams should standardize local QC and threshold policy before broader clinical-facing use.

---

*Last updated: 2026-04-19.*
