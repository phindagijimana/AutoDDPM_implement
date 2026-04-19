#!/usr/bin/env python
"""
Clean Inference Script for autoDDPM
====================================

Purpose: Run autoDDPM anomaly detection on custom T1w brain MRI volumes

Usage:
    python inference_clean.py --input brain.nii.gz --output results/
    
    Optional arguments:
        --model_path: Path to trained model (default: latest_model.pt)
        --noise_recon: Reconstruction noise level (default: 200)
        --noise_inpaint: Inpainting noise level (default: 50)
        --resample_steps: Number of resampling steps (default: 5)
        --threshold: Masking threshold (default: -1 for auto)
        --batch_size: Batch size for processing slices (default: 8)
        --device: Device to use (default: cuda if available)

Author: autoDDPM Implementation
Date: October 2025
"""

import os
import sys
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_zoo.ddpm import DDPM


class NIfTISliceDataset(Dataset):
    """Dataset for loading 3D MRI volume slice by slice."""
    
    def __init__(self, volume_data, target_size=(128, 128)):
        """
        Args:
            volume_data: 3D numpy array (H, W, D)
            target_size: Target size for each slice
        """
        self.volume_data = volume_data
        self.target_size = target_size
        self.num_slices = volume_data.shape[2]
        
    def __len__(self):
        return self.num_slices
    
    def __getitem__(self, idx):
        # Get slice along Z-axis (axial)
        slice_2d = self.volume_data[:, :, idx]
        
        # Skip empty slices
        if np.max(slice_2d) == 0:
            return torch.zeros(1, *self.target_size)
        
        # Normalize to [0, 1]
        slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
        
        # Convert to tensor and add channel dimension
        slice_tensor = torch.tensor(slice_norm, dtype=torch.float32).unsqueeze(0)
        
        # Resize
        transform = T.Resize(self.target_size, interpolation=T.InterpolationMode.BILINEAR)
        slice_resized = transform(slice_tensor)
        
        return slice_resized


def load_model(model_path, device):
    """Load trained DDPM model."""
    print(f"Loading model from: {model_path}")
    
    model = DDPM(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=[128, 256, 256],
        attention_levels=[False, True, True],
        num_res_blocks=1,
        num_head_channels=256,
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_weights"])
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded successfully on {device}")
    return model


def process_volume(model, volume_path, output_dir, args):
    """
    Process a full 3D volume through autoDDPM pipeline.
    
    Args:
        model: Trained DDPM model
        volume_path: Path to input NIfTI volume
        output_dir: Directory to save results
        args: Command line arguments
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(volume_path)}")
    print(f"{'='*60}\n")
    
    # Load NIfTI volume
    nifti_img = nib.load(volume_path)
    volume_data = nifti_img.get_fdata()
    print(f"Volume shape: {volume_data.shape}")
    
    # Initialize output volumes
    anomaly_map_volume = np.zeros_like(volume_data)
    reconstruction_volume = np.zeros_like(volume_data)
    
    # Create dataset and dataloader
    dataset = NIfTISliceDataset(volume_data, target_size=(128, 128))
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Process slices
    print(f"Processing {len(dataset)} slices with batch size {args.batch_size}...")
    z_idx = 0
    
    for batch in tqdm(dataloader, desc="Inference"):
        batch = batch.to(args.device)
        
        # Skip empty batches
        if torch.sum(batch) == 0:
            z_idx += batch.shape[0]
            continue
        
        # Run autoDDPM anomaly detection
        with torch.no_grad():
            anomaly_maps, anomaly_scores, x_rec_dict = model.get_anomaly(
                batch, 
                noise_level=args.noise_recon,
                method='autoDDPM',
                verbose=False
            )
        
        # Get reconstruction
        reconstruction = x_rec_dict['x_rec'].cpu()
        
        # Convert to numpy
        if isinstance(anomaly_maps, torch.Tensor):
            anomaly_maps = anomaly_maps.cpu().numpy()
        
        # Resize back to original dimensions
        resize_back = T.Resize((volume_data.shape[0], volume_data.shape[1]), 
                               interpolation=T.InterpolationMode.BILINEAR)
        
        # Process each slice in batch
        for i in range(batch.shape[0]):
            if z_idx + i >= volume_data.shape[2]:
                break
                
            # Resize anomaly map
            anomaly_slice = torch.tensor(anomaly_maps[i]).unsqueeze(0)
            anomaly_resized = resize_back(anomaly_slice).squeeze().numpy()
            
            # Resize reconstruction
            rec_slice = reconstruction[i]
            rec_resized = resize_back(rec_slice).squeeze().numpy()
            
            # Store in volumes
            anomaly_map_volume[:, :, z_idx + i] = anomaly_resized
            reconstruction_volume[:, :, z_idx + i] = rec_resized
        
        z_idx += batch.shape[0]
    
    # Save results
    save_results(
        anomaly_map_volume, 
        reconstruction_volume, 
        volume_data,
        nifti_img.affine, 
        output_dir,
        os.path.basename(volume_path)
    )
    
    print(f"\n✓ Processing complete! Results saved to: {output_dir}\n")


def save_results(anomaly_map, reconstruction, original, affine, output_dir, filename):
    """Save anomaly maps, reconstructions, and visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = filename.replace('.nii.gz', '').replace('.nii', '')
    
    print("\nSaving results...")
    
    # 1. Save full 3D volumes as NIfTI
    nib.save(
        nib.Nifti1Image(anomaly_map, affine=affine),
        os.path.join(output_dir, f"{base_name}_anomaly_map.nii.gz")
    )
    print(f"  ✓ Saved: {base_name}_anomaly_map.nii.gz")
    
    nib.save(
        nib.Nifti1Image(reconstruction, affine=affine),
        os.path.join(output_dir, f"{base_name}_reconstruction.nii.gz")
    )
    print(f"  ✓ Saved: {base_name}_reconstruction.nii.gz")
    
    # 2. Save binary threshold maps at different percentiles
    for percentile in [85, 90, 95]:
        threshold = np.percentile(anomaly_map, percentile)
        binary_map = (anomaly_map >= threshold).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(binary_map, affine=affine),
            os.path.join(output_dir, f"{base_name}_binary_{percentile}th.nii.gz")
        )
        print(f"  ✓ Saved: {base_name}_binary_{percentile}th.nii.gz")
    
    # 3. Save visualization of middle slice
    middle_idx = anomaly_map.shape[2] // 2
    visualize_slice(
        original[:, :, middle_idx],
        reconstruction[:, :, middle_idx],
        anomaly_map[:, :, middle_idx],
        output_dir,
        f"{base_name}_middle_slice.png"
    )
    print(f"  ✓ Saved: {base_name}_middle_slice.png")
    
    # 4. Save statistics
    stats = {
        'mean_anomaly_score': float(np.mean(anomaly_map)),
        'max_anomaly_score': float(np.max(anomaly_map)),
        'percentile_85': float(np.percentile(anomaly_map, 85)),
        'percentile_90': float(np.percentile(anomaly_map, 90)),
        'percentile_95': float(np.percentile(anomaly_map, 95)),
        'volume_shape': original.shape,
    }
    
    stats_path = os.path.join(output_dir, f"{base_name}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("autoDDPM Anomaly Detection Results\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"  ✓ Saved: {base_name}_stats.txt")


def visualize_slice(original, reconstruction, anomaly_map, output_dir, filename):
    """Create visualization comparing original, reconstruction, and anomaly map."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Slice', fontsize=14)
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction', fontsize=14)
    axes[1].axis('off')
    
    # Anomaly map
    im = axes[2].imshow(anomaly_map, cmap='hot')
    axes[2].set_title(f'Anomaly Map (max: {np.max(anomaly_map):.3f})', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='autoDDPM Inference on Custom Brain MRI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input NIfTI volume (.nii or .nii.gz)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--model_path', type=str, 
                        default='./latest_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--noise_recon', type=int, default=200,
                        help='Noise level for reconstruction (paper: 200)')
    parser.add_argument('--noise_inpaint', type=int, default=50,
                        help='Noise level for inpainting (paper: 50)')
    parser.add_argument('--resample_steps', type=int, default=5,
                        help='Number of resampling steps (paper: 5)')
    parser.add_argument('--threshold', type=float, default=0.13,
                        help='Masking threshold (paper: 0.13 for ATLAS, -1 for auto)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for slice processing')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    print("\n" + "="*60)
    print("autoDDPM Anomaly Detection - Clean Inference")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Noise (recon/inpaint): {args.noise_recon}/{args.noise_inpaint}")
    print(f"  Resample steps: {args.resample_steps}")
    print(f"  Batch size: {args.batch_size}")
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Update model parameters
    model.noise_level_recon = args.noise_recon
    model.noise_level_inpaint = args.noise_inpaint
    model.resample_steps = args.resample_steps
    model.masking_threshold = args.threshold
    
    # Process volume
    process_volume(model, args.input, args.output, args)
    
    print("="*60)
    print("✓ All done! Check output directory for results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


