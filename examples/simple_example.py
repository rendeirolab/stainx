#!/usr/bin/env python3
# Copyright (C) Rendeiro Group, CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences
# All rights reserved.
#
# This software is distributed under the terms of the GNU General Public License v3 (GPLv3).
# See the LICENSE file for details.

import argparse
import glob
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stainx import HistogramMatching, Macenko, Reinhard
from stainx.utils import ChannelFormatConverter

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


def tensor_to_numpy_image(tensor):
    """Convert tensor (C, H, W) or (1, C, H, W) to numpy array (H, W, C) for display."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present
    # Convert from (C, H, W) to (H, W, C)
    img = tensor.cpu().permute(1, 2, 0).numpy()
    # Clamp values to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0, 1)
    return img


def add_rounded_corners_pil(image_array, radius=75, border_thickness=100):
    img_array_uint8 = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
    pil_img = Image.fromarray(img_array_uint8)
    min_dimension = min(pil_img.width, pil_img.height)
    max_radius = min_dimension // 2
    practical_max_radius = min_dimension // 3
    if radius > max_radius:
        print(f"Warning: radius {radius} exceeds maximum {max_radius} for image size "
              f"{pil_img.width}x{pil_img.height}. Clamping to {max_radius}.")
        radius = max_radius
    elif radius > practical_max_radius:
        print(f"Note: radius {radius} is large. Consider using <= {practical_max_radius} "
              f"for better visual results.")
    new_width = pil_img.width + 2 * border_thickness
    new_height = pil_img.height + 2 * border_thickness
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    mask = Image.new('L', (new_width, new_height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([(radius, radius), (new_width - radius, new_height - radius)], fill=255)
    draw.rectangle([(radius, 0), (new_width - radius, radius)], fill=255)
    draw.rectangle([(radius, new_height - radius), (new_width - radius, new_height)], fill=255)
    draw.rectangle([(0, radius), (radius, new_height - radius)], fill=255)
    draw.rectangle([(new_width - radius, radius), (new_width, new_height - radius)], fill=255)
    draw.pieslice([(0, 0), (2*radius - 1, 2*radius - 1)], 180, 270, fill=255)
    draw.pieslice([(new_width - 2*radius, 0), (new_width - 1, 2*radius - 1)], 270, 360, fill=255)
    draw.pieslice([(new_width - 2*radius, new_height - 2*radius), (new_width - 1, new_height - 1)], 0, 90, fill=255)
    draw.pieslice([(0, new_height - 2*radius), (2*radius - 1, new_height - 1)], 90, 180, fill=255)
    paste_x = border_thickness
    paste_y = border_thickness
    new_image.paste(pil_img, (paste_x, paste_y))
    new_image_rgba = new_image.convert('RGBA')
    alpha = mask.copy()
    final = Image.new('RGB', new_image.size, (255, 255, 255))
    final.paste(new_image_rgba, mask=alpha)
    result = np.array(final).astype(np.float32) / 255.0
    return result


def visualize_results(reference_image, source_images, transformed_images, test_paths, method, save_plots=False, output_dir=None):
    """Visualize reference, original, and transformed images."""
    num_images = len(test_paths)
    
    # Convert tensors to numpy arrays for visualization
    ref_img = tensor_to_numpy_image(reference_image)
    
    # Add rounded corners with white borders
    radius = 20  # Corner radius in pixels
    border_thickness = 5  # White border thickness in pixels
    ref_img = add_rounded_corners_pil(ref_img, radius=radius, border_thickness=border_thickness)
    
    # Create figure with subplots: reference at top spanning full width, then original and transformed side by side
    fig = plt.figure(figsize=(14, 3 + 2.5 * num_images), facecolor='white')
    gs = fig.add_gridspec(num_images + 1, 2, hspace=0.3, wspace=0.1)
    
    # Plot reference image at the top spanning both columns
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(ref_img)
    ax.set_title("Reference (Target) Image", fontsize=18, fontweight='bold')
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Plot original and transformed images side by side
    for i in range(num_images):
        orig_img = tensor_to_numpy_image(source_images[i])
        trans_img = tensor_to_numpy_image(transformed_images[i])
        
        # Add rounded corners with white borders
        orig_img = add_rounded_corners_pil(orig_img, radius=radius, border_thickness=border_thickness)
        trans_img = add_rounded_corners_pil(trans_img, radius=radius, border_thickness=border_thickness)
        
        # Original image
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(orig_img)
        if i == 0:
            ax.set_title("Original", fontsize=18, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('white')
        
        # Transformed image
        ax = fig.add_subplot(gs[i + 1, 1])
        ax.imshow(trans_img)
        if i == 0:
            ax.set_title(f"Transformed\n{method}", fontsize=18, fontweight='bold')
        ax.axis('off')
        ax.set_facecolor('white')
        
    plt.tight_layout()
    
    if save_plots:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "data", "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"normalization_results_{method}.pdf")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
        print(f"\nSaved visualization to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark for a single StainX method")
    parser.add_argument("method", type=str, choices=["reinhard", "macenko", "histogram_matching"], help="StainX method to benchmark")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing images (default: examples/data)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to use")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--visualize", action="store_true", help="Display visualization of results")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots to files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots (default: examples/data/output)")

    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    else:
        data_dir = args.data_dir

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'MPS' if device == 'mps' else 'CPU'}")
    print(f"Method: {args.method}")
    print(f"Data directory: {data_dir}")
    print()

    # Load target image (reference for fitting)
    target_path = os.path.join(data_dir, "target.png")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target image not found: {target_path}")
    
    print(f"Loading target image: {target_path}")
    target_img = Image.open(target_path).convert("RGB")
    target_tensor = transforms.ToTensor()(target_img).unsqueeze(0)  # (1, 3, H, W)
    reference_image = target_tensor.to(device)
    
    # Load test images (source images to transform)
    test_images = []
    # Find all PNG files in data directory, excluding target.png
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    test_paths = []
    for ext in image_extensions:
        test_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    
    # Filter out target.png and sort for consistent ordering
    test_paths = [p for p in test_paths if os.path.basename(p) != "target.png"]
    test_paths.sort()
    
    if not test_paths:
        raise FileNotFoundError(f"No test images found in {data_dir} (excluding target.png)")
    
    print(f"Found {len(test_paths)} test image(s)")
    for i, test_path in enumerate(test_paths, 1):
        print(f"Loading test image {i}/{len(test_paths)}: {os.path.basename(test_path)}")
        test_img = Image.open(test_path).convert("RGB")
        test_tensor = transforms.ToTensor()(test_img)  # (3, H, W)
        test_images.append(test_tensor)
    
    # Stack test images into a batch
    source_image = torch.stack(test_images).to(device)  # (N, 3, H, W)
    
    print(f"Reference image shape: {reference_image.shape}")
    print(f"Source images shape: {source_image.shape}")
    print()

    # Create normalizer based on method
    if args.method == "reinhard":
        normalizer = Reinhard(device=device)
    elif args.method == "macenko":
        normalizer = Macenko(device=device)
    elif args.method == "histogram_matching":
        converter = ChannelFormatConverter(channel_axis=1)
        reference_image = converter.prepare_for_normalizer(reference_image)
        source_image = converter.prepare_for_normalizer(source_image)
        normalizer = HistogramMatching(device=device, channel_axis=1)

    # Fit on reference image
    print("Fitting normalizer...")
    normalizer.fit(reference_image)

    # Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()

    # Run transform once
    print("Running transform...")
    start_time = time.time()
    for _ in range(args.runs):
        result = normalizer.transform(source_image)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_time = ((time.time() - start_time) * 1000) / args.runs

    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    print(f"Time: {elapsed_time:.3f} ms")
    print(f"Images per second: {source_image.shape[0] * 1000 / elapsed_time:.2f}")
    
    # Visualize results if requested
    if args.visualize or args.save_plots:
        print("\nGenerating visualization...")
        # Move result back to CPU for visualization if needed
        result_cpu = result.cpu() if result.is_cuda else result
        source_cpu = source_image.cpu() if source_image.is_cuda else source_image
        reference_cpu = reference_image.cpu() if reference_image.is_cuda else reference_image
        
        visualize_results(
            reference_cpu,
            source_cpu,
            result_cpu,
            test_paths,
            args.method,
            save_plots=args.save_plots,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
