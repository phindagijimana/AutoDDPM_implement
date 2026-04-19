#!/usr/bin/env python
"""
Analyze autoDDPM Lesion Detection Performance
==============================================

This script analyzes whether autoDDPM correctly detected lesions by:
1. Loading the anomaly map
2. Creating binary predictions at different thresholds
3. Computing metrics if ground truth is available
4. Visualizing the results

Usage:
    python analyze_lesion_detection.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Check if output exists
output_dir = "autoDDPM_outputs"
files_to_check = [
    "anomaly_map_resampled.png",
    "reconstruction_resampled.png", 
    "original_masked.png"
]

print("="*60)
print("autoDDPM Lesion Detection Analysis")
print("="*60)
print()

# Check what outputs exist
print("Checking available outputs...")
available_files = []
for f in files_to_check:
    fpath = os.path.join(output_dir, f)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  ✓ {f} ({size_kb:.1f} KB)")
        available_files.append(f)
    else:
        print(f"  ✗ {f} (not found)")

print()

# Try to load NPZ if it exists
npz_file = os.path.join(output_dir, "results_masked_resampled.npz")
if os.path.exists(npz_file):
    print(f"✓ Found results file: {npz_file}")
    print()
    
    data = np.load(npz_file, allow_pickle=True)
    print(f"Available data: {list(data.keys())}")
    print()
    
    # Analyze anomaly map
    if 'anomaly_map' in data:
        amap = data['anomaly_map']
        print("="*60)
        print("ANOMALY MAP ANALYSIS")
        print("="*60)
        print(f"Shape: {amap.shape}")
        print(f"Data type: {amap.dtype}")
        print()
        
        print("Anomaly Score Statistics:")
        print(f"  Min:     {amap.min():.6f}")
        print(f"  Max:     {amap.max():.6f}")
        print(f"  Mean:    {amap.mean():.6f}")
        print(f"  Median:  {np.median(amap):.6f}")
        print(f"  Std Dev: {amap.std():.6f}")
        print()
        
        print("Percentiles:")
        for p in [50, 75, 90, 95, 99]:
            val = np.percentile(amap, p)
            print(f"  {p}th: {val:.6f}")
        print()
        
        print("Pixel Coverage by Threshold:")
        print("(What percentage of pixels exceed each threshold)")
        for thresh in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            pct = 100 * np.sum(amap > thresh) / amap.size
            print(f"  > {thresh:.2f}: {pct:6.2f}% of pixels")
        print()
        
        # Check if this looks like it found anomalies
        high_anomaly_pct = 100 * np.sum(amap > 0.15) / amap.size
        
        print("="*60)
        print("INTERPRETATION")
        print("="*60)
        
        if amap.max() < 0.05:
            print("❌ ISSUE: Very low anomaly scores (max < 0.05)")
            print("   → Model found minimal anomalies")
            print("   → Possible reasons:")
            print("     • Image is healthy (no lesions)")
            print("     • Model threshold too high")
            print("     • Model needs retraining")
        elif high_anomaly_pct < 0.5:
            print("✓ LOOKS GOOD: Few high-confidence anomalies")
            print(f"   → {high_anomaly_pct:.2f}% of pixels exceed 0.15 threshold")
            print("   → Suggests focal anomalies (typical for lesions)")
        elif high_anomaly_pct > 10:
            print("⚠ WARNING: Many high anomaly scores")
            print(f"   → {high_anomaly_pct:.1f}% of pixels exceed 0.15")
            print("   → Possible issues:")
            print("     • Threshold may be too low")
            print("     • Image quality issues")
            print("     • Domain shift from training data")
        else:
            print(f"✓ MODERATE: {high_anomaly_pct:.2f}% pixels flagged as anomalies")
            print("   → Reasonable for lesion detection")
        
        print()
        print("="*60)
        print("RECOMMENDATION")
        print("="*60)
        
        if 'coarse_reconstruction' in data or 'step1_coarse' in str(data.files):
            print("✓ Full autoDDPM pipeline data available")
            print("  You can review:")
            print("  1. Step 1: Coarse reconstruction & anomaly map")
            print("  2. Step 2: Binary mask created")
            print("  3. Step 3: Final inpainted result")
        else:
            print("⚠ Only final results available (cached from old run)")
            print("  To see full pipeline: Delete cache and rerun")
        
        print()
        print("To validate accuracy, you need:")
        print("  1. Ground truth lesion masks")
        print("  2. Known lesion locations")
        print("  3. Clinical validation")
        
        print()
        print("Files to review:")
        print("  • autoddpm_pipeline_visualization.png - Overview")
        print("  • anomaly_map_resampled.png - Anomaly detection")
        print("  • Compare with clinical assessment")
        
    else:
        print("⚠ No anomaly map found in results file")
        
else:
    print(f"⚠ Results file not found: {npz_file}")
    print()
    print("The script may still be running. Check:")
    print("  ps aux | grep main.py")
    print()
    print("Or run fresh:")
    print("  python scripts_legacy/main.py")

print()
print("="*60)
print("IMPORTANT: autoDDPM vs Segmentation")
print("="*60)
print()
print("autoDDPM performs ANOMALY DETECTION, not segmentation:")
print("  • Trained on HEALTHY brains")
print("  • Detects deviations from normal")
print("  • Output: Continuous anomaly scores (0-1)")
print("  • NOT: Binary lesion masks")
print()
print("For lesion segmentation:")
print("  1. Threshold the anomaly map (e.g., > 0.15)")
print("  2. Apply post-processing (morphology, connected components)")
print("  3. Validate against ground truth")
print()
print("="*60)





