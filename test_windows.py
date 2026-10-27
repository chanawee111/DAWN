"""
Cross-platform steganography testing script (Windows/Linux).

Before running:
1. Configure dataset_path, result_folder, model_epoch, and other model settings
   in utilities/load_test_setting.py.
2. Ensure the checkpoint exists at:
   <result_folder>/models/Encoder_<model_epoch>.pth
3. Run this script from the project root so imports such as utilities and models work.
"""

import os
import csv
import re
from pathlib import Path

import numpy as np
import torch
import kornia
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision

from utilities import *
from models.model import *
from utilities.load_test_setting import *
from models.noise_layers import JpegARWGAN420_v2


# ------------------------------------------------
# Utility functions
# ------------------------------------------------

def denormalize(tensor):
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    tensor = tensor * 255
    return tensor.byte()


def normalize(tensor):
    tensor = tensor.float()
    tensor = tensor / 255.0
    tensor = tensor * 2.0
    tensor = tensor - 1.0
    return tensor


def plot_metric_distribution(data, metric_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.histplot(data, kde=True, stat="density")
    plt.title(f"{metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = output_dir / f"{metric_name}_distribution.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_flat_ratio_comparison(filtered_results, output_dir):
    """Plot metrics across different flat-color-ratio ranges."""
    ranges = sorted(filtered_results.keys())
    metrics = ["BER", "Accuracy", "PSNR", "LPIPS", "SSIM"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    range_labels = list(ranges)
    x_pos = np.arange(len(ranges))

    metric_key_map = {
        "BER": ("avg_ber", "std_ber"),
        "Accuracy": ("avg_accuracy", "std_accuracy"),
        "PSNR": ("avg_psnr", "std_psnr"),
        "LPIPS": ("avg_lpips", "std_lpips"),
        "SSIM": ("avg_ssim", "std_ssim"),
    }

    for idx, metric in enumerate(metrics):
        avg_key, std_key = metric_key_map[metric]
        means = [filtered_results[r]["stats"][avg_key] for r in ranges]
        stds = [filtered_results[r]["stats"][std_key] for r in ranges]

        axes[idx].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(range_labels, rotation=45)
        axes[idx].set_xlabel("Flat Color Ratio Range")
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f"{metric} by Flat Color Ratio")
        axes[idx].grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.delaxes(axes[5])
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "flat_ratio_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()


def get_encoder_decoder(model):
    """Return the underlying encoder-decoder for DataParallel or regular models."""
    encoder_decoder = model.encoder_decoder
    if isinstance(encoder_decoder, torch.nn.DataParallel):
        encoder_decoder = encoder_decoder.module
    return encoder_decoder


def run_test(
    model,
    test_dataloader,
    device,
    message_length,
    desc="Testing",
    jpeg_sim_instance=None,
    save_images_number=4,
):
    """
    Run testing over a dataloader.
    Returns per-batch results, aggregate statistics, metric distributions,
    and sample images.
    """
    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    lpips_loss.eval()

    total_ber, total_acc, total_psnr, total_lpips, total_ssim = [], [], [], [], []
    batch_results = []
    processed_images = 0

    total_batches = len(test_dataloader)
    if total_batches > 0:
        saved_iterations = set(
            np.random.choice(
                np.arange(total_batches),
                size=min(save_images_number, total_batches),
                replace=False,
            ).tolist()
        )
    else:
        saved_iterations = set()

    saved_original_images = []
    saved_encoded_images = []

    encoder_decoder = get_encoder_decoder(model)

    model.encoder_decoder.eval()
    if hasattr(model, "discriminator"):
        model.discriminator.eval()

    for i, images in enumerate(tqdm(test_dataloader, desc=desc)):
        image = images.to(device)
        jpeg_diff_map = compute_jpeg_diff_map_differentiable(
            image, jpeg_sim_instance
        )
        processed_images += image.size(0)

        message = torch.randint(
            0, 2, (image.size(0), message_length), device=device
        ).float()

        with torch.no_grad():
            encoded_images, _, _, _ = encoder_decoder.encoder(
                image, message, jpeg_diff_map
            )

            # Retained from the original script: convert through 8-bit values.
            encoded_images_denormalized = denormalize(encoded_images)
            encoded_images_denormalized = normalize(encoded_images_denormalized)

            noised_images = encoder_decoder.noise([encoded_images, image])
            decoded_messages = torch.sigmoid(
                encoder_decoder.decoder(noised_images)
            )

            # Metrics are calculated on the original normalized tensors.
            psnr = kornia.metrics.psnr(encoded_images, image, 2.0).item()
            lpips = lpips_loss(encoded_images, image).item()
            ssim = kornia.metrics.ssim(
                encoded_images, image, window_size=5
            ).mean().item()

            error_rate = model.decoded_message_error_rate_batch(
                message, decoded_messages
            )
            accuracy = model.decoded_message_accuracy_batch(
                message, decoded_messages
            )
            ber = model.calculate_BER_batch(message, decoded_messages)

        total_ber.append(ber)
        total_acc.append(accuracy)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
        total_lpips.append(lpips)

        batch_results.append(
            {
                "batch_index": i,
                "batch_size": image.size(0),
                "accuracy": accuracy,
                "error_rate": error_rate,
                "BER": ber,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
            }
        )

        if i in saved_iterations:
            saved_original_images.append(image.detach().cpu())
            saved_encoded_images.append(encoded_images.detach().cpu())

    if total_ber:
        stats = {
            "avg_ber": np.mean(total_ber),
            "std_ber": np.std(total_ber),
            "avg_accuracy": np.mean(total_acc),
            "std_accuracy": np.std(total_acc),
            "avg_psnr": np.mean(total_psnr),
            "std_psnr": np.std(total_psnr),
            "avg_lpips": np.mean(total_lpips),
            "std_lpips": np.std(total_lpips),
            "avg_ssim": np.mean(total_ssim),
            "std_ssim": np.std(total_ssim),
            "processed_images": processed_images,
        }
    else:
        stats = {
            key: 0
            for key in [
                "avg_ber", "std_ber",
                "avg_accuracy", "std_accuracy",
                "avg_psnr", "std_psnr",
                "avg_lpips", "std_lpips",
                "avg_ssim", "std_ssim",
                "processed_images",
            ]
        }

    return {
        "batch_results": batch_results,
        "stats": stats,
        "distributions": {
            "ber": total_ber,
            "accuracy": total_acc,
            "psnr": total_psnr,
            "lpips": total_lpips,
            "ssim": total_ssim,
        },
        "original_images": saved_original_images,
        "encoded_images": saved_encoded_images,
    }


def save_range_results(results, range_label, output_dir):
    """Save per-batch CSV, summary text, metric plots, and image grids."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = results["stats"]

    # Per-batch CSV
    per_batch_csv = output_dir / "per_batch_results.csv"
    with per_batch_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "batch_index", "batch_size", "accuracy", "error_rate",
                "BER", "psnr", "ssim", "lpips",
            ]
        )
        for batch in results["batch_results"]:
            writer.writerow(
                [
                    batch["batch_index"],
                    batch["batch_size"],
                    batch["accuracy"],
                    batch["error_rate"],
                    batch["BER"],
                    batch["psnr"],
                    batch["ssim"],
                    batch["lpips"],
                ]
            )

    # Summary text
    summary_txt = output_dir / "results_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("TEST SUMMARY\n")
        f.write(f"Processed Images: {stats['processed_images']}\n\n")
        f.write(f"BER:      {stats['avg_ber']:.4f} ± {stats['std_ber']:.4f}\n")
        f.write(
            f"Accuracy: {stats['avg_accuracy']:.4f} ± "
            f"{stats['std_accuracy']:.4f}\n"
        )
        f.write(f"PSNR:     {stats['avg_psnr']:.2f} ± {stats['std_psnr']:.2f}\n")
        f.write(f"LPIPS:    {stats['avg_lpips']:.4f} ± {stats['std_lpips']:.4f}\n")
        f.write(f"SSIM:     {stats['avg_ssim']:.4f} ± {stats['std_ssim']:.4f}\n")

    # Metric distribution plots
    for metric_name, data in results["distributions"].items():
        if len(data) > 0:
            plot_metric_distribution(data, metric_name.upper(), output_dir)

    # Save image grids
    original_images = results.get("original_images", [])
    encoded_images = results.get("encoded_images", [])

    if original_images and encoded_images:
        original_all = torch.cat(original_images, dim=0)
        encoded_all = torch.cat(encoded_images, dim=0)

        original_grid = torchvision.utils.make_grid(
            original_all, nrow=4, normalize=True, scale_each=True
        )
        torchvision.utils.save_image(
            original_grid, output_dir / "grid_original_images.png"
        )

        encoded_grid = torchvision.utils.make_grid(
            encoded_all, nrow=4, normalize=True, scale_each=True
        )
        torchvision.utils.save_image(
            encoded_grid, output_dir / "grid_encoded_images.png"
        )


# ------------------------------------------------
# Configuration and testing
# ------------------------------------------------

def main():
    # Initialize logging (creates test result files).
    initialize_test_logging()

    jpeg_quality = 50
    jpeg_subsample = 2  # PIL: 2 = JPEG 4:2:0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jpeg_sim_layer = JpegARWGAN420_v2(quality=jpeg_quality)

    print("Loading test settings...")
    print("-----------------------------------")

    # pathlib makes path handling work on Windows and Linux.
    experiment_dir = Path(result_folder)
    dataset_root = Path(dataset_path)

    print(f"Experiment folder: {experiment_dir}")
    match = re.search(r"(exp\d+)", str(experiment_dir))
    result_name = match.group(1) if match else "default_exp"

    output_dir = (
        Path(".")
        / "Test_results"
        / f"results_run({result_name})_epoch{model_epoch}"
        / f"JpegTest_420_jpeg{jpeg_quality}_test5000_jpeg{strength_factor}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTest dataset path: {dataset_root}")
    print(f"Device: {device}")

    # Check dataset path before model initialization.
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{dataset_root}\n\n"
            "Update dataset_path in utilities/load_test_setting.py."
        )

    # Model
    noise_layers_alt = [f"JpegTest({jpeg_quality}, subsample={jpeg_subsample})"]

    model = Model(
        H,
        W,
        message_length,
        noise_layers_alt,
        device,
        batch_size,
        lr,
        decoder_only_training=decoder_only_training,
    )

    encoder_path = (
        experiment_dir / "models" / f"Encoder_{model_epoch}.pth"
    )
    if not encoder_path.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint not found:\n{encoder_path}\n"
            "Check result_folder and model_epoch in load_test_setting.py."
        )

    print(f"Loading encoder checkpoint: {encoder_path}")
    model.load_model_ed(str(encoder_path))

    # Main loop: flat dataset
    print("\n" + "=" * 60)
    print("TESTING PIPELINE — FLAT TEST DATASET")
    print("=" * 60)
    print(f"\n[Testing] -> {dataset_root}")

    dataset = Dataset_loader(str(dataset_root), H, W)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Safe default for Windows.
        pin_memory=(device.type == "cuda"),
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found in {dataset_root}. "
            "Check Dataset_loader and supported file extensions."
        )

    print(f"Images found by Dataset_loader: {len(dataset)}")

    results = run_test(
        model,
        dataloader,
        device,
        message_length,
        desc="Testing",
        jpeg_sim_instance=jpeg_sim_layer,
        save_images_number=save_images_number,
    )

    save_range_results(results, "test_5000", output_dir)

    stats = results["stats"]
    print("\nTesting completed!")
    print(f"Processed Images: {stats['processed_images']}")
    print(f"BER:      {stats['avg_ber']:.4f} ± {stats['std_ber']:.4f}")
    print(
        f"Accuracy: {stats['avg_accuracy']:.4f} ± "
        f"{stats['std_accuracy']:.4f}"
    )
    print(f"PSNR:     {stats['avg_psnr']:.2f} ± {stats['std_psnr']:.2f}")
    print(f"SSIM:     {stats['avg_ssim']:.4f} ± {stats['std_ssim']:.4f}")
    print(f"LPIPS:    {stats['avg_lpips']:.4f} ± {stats['std_lpips']:.4f}")

    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.resolve()}")
    print("Results saved:")
    print("  - per_batch_results.csv")
    print("  - results_summary.txt")
    print("  - grid_original_images.png (if sample images were saved)")
    print("  - grid_encoded_images.png (if sample images were saved)")
    print("  - BER_distribution.png")
    print("  - ACCURACY_distribution.png")
    print("  - PSNR_distribution.png")
    print("  - LPIPS_distribution.png")
    print("  - SSIM_distribution.png")


if __name__ == "__main__":
    main()
