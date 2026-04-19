import argparse
import torch
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # No Ensure GUI backend for plotting
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import os
import sys

# Add autoDDPM project folder to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level to project root
sys.path.insert(0, project_root)


from model_zoo.ddpm import DDPM  # Your AutoDDPM class


def _nifti_stem(path: str) -> str:
    b = os.path.basename(path)
    if b.endswith(".nii.gz"):
        return b[: -len(".nii.gz")]
    if b.endswith(".nii"):
        return b[: -len(".nii")]
    return os.path.splitext(b)[0]


parser = argparse.ArgumentParser(description="autoDDPM single-slice TRACK demo")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Input NIfTI (default: TRACK_Sample_Image/sample_T1w_2.nii.gz under project root)",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Delete cached npz for this volume and re-run inference",
)
cli = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load trained DDPM model with PAPER PARAMETERS
# -------------------------------
model = DDPM(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=[128, 256, 256],
    attention_levels=[False, True, True],
    num_res_blocks=1,
    num_head_channels=256,
    # CRITICAL: Use parameters from the paper (autoddpm.yaml)
    noise_level_recon=200,      # ✅ Paper value (not 300!)
    noise_level_inpaint=50,     # ✅ Paper value
    resample_steps=5,           # ✅ Paper value (not 4!)
    masking_threshold=0.13,     # ✅ Paper value for ATLAS data
    method="autoDDPM",
)
checkpoint_path = os.path.join(project_root, "latest_model.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_weights"])
model.eval().to(device)

# -------------------------------
# Define output directory
# -------------------------------
output_dir = "autoDDPM_outputs"
os.makedirs(output_dir, exist_ok=True)

default_img = os.path.join(project_root, "TRACK_Sample_Image", "sample_T1w_2.nii.gz")
img_path = os.path.abspath(cli.input if cli.input is not None else default_img)
if not os.path.isfile(img_path):
    print(f"[ERROR] Input not found: {img_path}")
    sys.exit(1)

results_path = os.path.join(
    output_dir, f"results_masked_resampled_{_nifti_stem(img_path)}.npz"
)
if cli.force and os.path.isfile(results_path):
    os.remove(results_path)

# -------------------------------
# Load and preprocess image
# -------------------------------
img = nib.load(img_path)
img_data = img.get_fdata()

# Select axial middle slice and normalize
slice_ = img_data[:, :, img_data.shape[2] // 2]
slice_norm = (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_))
slice_norm = np.clip(slice_norm, 0, 1)
slice_tensor = torch.tensor(slice_norm, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

# Resize and add batch/channel dimensions: [1, 1, 256, 256]
transform = T.Resize((256, 256))
input_tensor = transform(slice_tensor.unsqueeze(0)).to(device)

# -------------------------------
# FIXED: Proper autoDDPM Implementation
# -------------------------------
# autoDDPM does its own masking internally (Mask, Stitch, Re-Sample)
# No need for manual masking!

mask_used = None
stitched = None

if os.path.exists(results_path):
    print("[INFO] Loading cached results...")
    data = np.load(results_path)
    original_img = data["original"]
    reconstructed_img = data["reconstruction"]
    anomaly_map_img = data["anomaly_map"]
    coarse_rec = data.get("coarse_reconstruction", None)
    coarse_map = data.get("coarse_anomaly", None)
    mask_used = data.get("binary_mask", None)
    stitched = data.get("stitched", None)
else:
    print("[INFO] Running autoDDPM inference (Mask, Stitch, Re-Sample)...")
    
    # Call autoDDPM - it handles everything internally
    with torch.no_grad():
        anomaly_maps, anomaly_scores, x_rec_dict = model.get_anomaly(
            input_tensor, 
            noise_level=model.noise_level_recon,  # Default: 200
            method='autoDDPM',  # Explicitly specify autoDDPM method
            verbose=False
        )
    
    # Extract results
    original_img = input_tensor[0, 0].detach().cpu().numpy()
    reconstructed_img = x_rec_dict['x_rec'][0, 0].detach().cpu().numpy()
    anomaly_map_img = anomaly_maps[0, 0] if isinstance(anomaly_maps, torch.Tensor) else anomaly_maps[0, 0]
    
    # autoDDPM also provides intermediate results:
    coarse_rec = x_rec_dict.get('x_rec_orig', None)  # Step 1: Initial reconstruction
    coarse_map = x_rec_dict.get('x_res_orig', None)  # Step 1: Initial anomaly map
    mask_used = x_rec_dict.get("mask", None)  # Step 2: Binary mask created
    stitched = x_rec_dict.get("stitch", None)  # Step 2: Stitched image

    if coarse_rec is not None:
        coarse_rec = coarse_rec[0, 0].detach().cpu().numpy()
    if coarse_map is not None:
        coarse_map = coarse_map[0, 0].detach().cpu().numpy()
    if mask_used is not None:
        mask_used = mask_used[0, 0].detach().cpu().numpy()
    if stitched is not None:
        stitched = stitched[0, 0].detach().cpu().numpy()

    # Save compressed result with intermediate steps
    save_dict = {
        'original': original_img,
        'reconstruction': reconstructed_img,
        'anomaly_map': anomaly_map_img,
    }
    if coarse_rec is not None:
        save_dict['coarse_reconstruction'] = coarse_rec
    if coarse_map is not None:
        save_dict['coarse_anomaly'] = coarse_map
    if mask_used is not None:
        save_dict['binary_mask'] = mask_used
    if stitched is not None:
        save_dict['stitched'] = stitched
    
    np.savez_compressed(results_path, **save_dict)

    # Save PNGs
    plt.imsave(os.path.join(output_dir, "original.png"), original_img, cmap='gray')
    plt.imsave(os.path.join(output_dir, "reconstruction_final.png"), reconstructed_img, cmap='gray')
    plt.imsave(os.path.join(output_dir, "anomaly_map_final.png"), anomaly_map_img, cmap='hot')
    
    # Save intermediate steps if available
    if coarse_rec is not None:
        plt.imsave(os.path.join(output_dir, "step1_coarse_reconstruction.png"), coarse_rec, cmap='gray')
    if coarse_map is not None:
        plt.imsave(os.path.join(output_dir, "step1_coarse_anomaly.png"), coarse_map, cmap='hot')
    if mask_used is not None:
        plt.imsave(os.path.join(output_dir, "step2_binary_mask.png"), mask_used, cmap='gray')
    if stitched is not None:
        plt.imsave(os.path.join(output_dir, "step2_stitched.png"), stitched, cmap='gray')

# -------------------------------
# Step 4: Visualize autoDDPM Results
# -------------------------------
# Show the 3-step autoDDPM pipeline
num_plots = 3 if coarse_rec is None else 6

if num_plots == 6:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: autoDDPM 3-step process
    axes[0, 0].imshow(coarse_rec, cmap='gray')
    axes[0, 0].set_title("Step 1: Coarse Reconstruction")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(coarse_map, cmap='hot')
    axes[0, 1].set_title("Step 1: Coarse Anomaly Map")
    axes[0, 1].axis('off')
    
    if mask_used is not None:
        axes[0, 2].imshow(mask_used, cmap='gray')
        axes[0, 2].set_title("Step 2: Binary Mask")
        axes[0, 2].axis('off')
    else:
        axes[0, 2].axis('off')
    
    # Bottom row: Final results
    axes[1, 0].imshow(original_img, cmap='gray')
    axes[1, 0].set_title("Original Input")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reconstructed_img, cmap='gray')
    axes[1, 1].set_title("Step 3: Final Inpainted")
    axes[1, 1].axis('off')
    
    im = axes[1, 2].imshow(anomaly_map_img, cmap='hot')
    axes[1, 2].set_title(f"Final Anomaly Map\nMax: {np.max(anomaly_map_img):.3f}")
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.suptitle("autoDDPM Pipeline: Mask → Stitch → Re-Sample", fontsize=16, fontweight='bold')
else:
    # Simple 3-panel view
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title("autoDDPM Reconstruction")
    axes[1].axis('off')
    
    im = axes[2].imshow(anomaly_map_img, cmap='hot')
    axes[2].set_title(f"Anomaly Map\nMax: {np.max(anomaly_map_img):.3f}")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "autoddpm_pipeline_visualization.png"), dpi=150, bbox_inches='tight')
print(f"[INFO] Results saved to: {output_dir}")
print(f"[INFO] Visualization: {os.path.join(output_dir, 'autoddpm_pipeline_visualization.png')}")
plt.close()
