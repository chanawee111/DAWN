import os
import csv
import re
import numpy as np
import torch
import kornia
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.figure(figsize=(8, 6))
    sns.histplot(data, kde=True, stat="density")
    plt.title(f"{metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path = os.path.join(output_dir, f"{metric_name}_distribution.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_flat_ratio_comparison(filtered_results, output_dir):
    """Plot metrics across different flat color ratio ranges"""
    ranges = sorted(filtered_results.keys())
    metrics = ['BER', 'Accuracy', 'PSNR', 'LPIPS', 'SSIM']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    range_labels = [r for r in ranges]  # Already strings like "0.00-0.20"
    x_pos = np.arange(len(ranges))

    metric_key_map = {
        'BER':      ('avg_ber',      'std_ber'),
        'Accuracy': ('avg_accuracy', 'std_accuracy'),
        'PSNR':     ('avg_psnr',     'std_psnr'),
        'LPIPS':    ('avg_lpips',    'std_lpips'),
        'SSIM':     ('avg_ssim',     'std_ssim'),
    }

    for idx, metric in enumerate(metrics):
        avg_key, std_key = metric_key_map[metric]
        means = [filtered_results[r]['stats'][avg_key] for r in ranges]
        stds  = [filtered_results[r]['stats'][std_key] for r in ranges]

        axes[idx].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(range_labels, rotation=45)
        axes[idx].set_xlabel('Flat Color Ratio Range')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} by Flat Color Ratio')
        axes[idx].grid(True, linestyle='--', alpha=0.3, axis='y')

    fig.delaxes(axes[5])
    plt.tight_layout()
    save_path = os.path.join(output_dir, "flat_ratio_comparison.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def run_test(model, test_dataloader, device, message_length, desc="Testing", jpeg_sim_instance=None):
    """
    Run testing loop on a dataloader (no filtering — data is pre-split by directory).
    Returns per-batch results, aggregate stats, and metric distributions.
    """
    LPIPS_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    total_ber, total_acc, total_psnr, total_lpips, total_ssim = [], [], [], [], []
    batch_results = []
    processed_images = 0

    for i, images in enumerate(tqdm(test_dataloader, desc=desc)):
        image = images.to(device)

        jpeg_diff_map = compute_jpeg_diff_map_differentiable(image, jpeg_sim_instance)
        processed_images += image.size(0)

        message = torch.randint(0, 2, (image.size(0), message_length)).float().to(device)

        model.encoder_decoder.eval()
        model.discriminator.eval()

        with torch.no_grad():
            encoded_images, _, _, _ = model.encoder_decoder.module.encoder(image, message,jpeg_diff_map)

            encoded_images_denormalized = denormalize(encoded_images)
            encoded_images_denormalized = normalize(encoded_images_denormalized)
            noised_images = model.encoder_decoder.module.noise(
                [encoded_images, images]
            )

            decoded_messages = torch.sigmoid(
                model.encoder_decoder.module.decoder(noised_images)
            )

            psnr  = kornia.metrics.psnr(encoded_images, image, 2.0).item()
            lpips = LPIPS_loss(encoded_images, image).item()
            ssim  = kornia.metrics.ssim(encoded_images, image, window_size=5).mean().item()

        error_rate = model.decoded_message_error_rate_batch(message, decoded_messages)
        accuracy   = model.decoded_message_accuracy_batch(message, decoded_messages)
        BER        = model.calculate_BER_batch(message, decoded_messages)

        total_ber.append(BER)
        total_acc.append(accuracy)
        total_psnr.append(psnr)
        total_ssim.append(ssim)
        total_lpips.append(lpips)

        batch_results.append({
            'batch_index': i,
            'batch_size':  image.size(0),
            'accuracy':    accuracy,
            'error_rate':  error_rate,
            'BER':         BER,
            'psnr':        psnr,
            'ssim':        ssim,
            'lpips':       lpips,
        })

    if len(total_ber) > 0:
        stats = {
            'avg_ber':      np.mean(total_ber),
            'std_ber':      np.std(total_ber),
            'avg_accuracy': np.mean(total_acc),
            'std_accuracy': np.std(total_acc),
            'avg_psnr':     np.mean(total_psnr),
            'std_psnr':     np.std(total_psnr),
            'avg_lpips':    np.mean(total_lpips),
            'std_lpips':    np.std(total_lpips),
            'avg_ssim':     np.mean(total_ssim),
            'std_ssim':     np.std(total_ssim),
            'processed_images': processed_images,
        }
    else:
        stats = {k: 0 for k in [
            'avg_ber', 'std_ber', 'avg_accuracy', 'std_accuracy',
            'avg_psnr', 'std_psnr', 'avg_lpips', 'std_lpips',
            'avg_ssim', 'std_ssim', 'processed_images'
        ]}

    return {
        'batch_results': batch_results,
        'stats': stats,
        'distributions': {
            'ber':      total_ber,
            'accuracy': total_acc,
            'psnr':     total_psnr,
            'lpips':    total_lpips,
            'ssim':     total_ssim,
        }
    }

def save_range_results(results, range_label, output_dir):
    """Save per-batch CSV, summary txt, and metric distribution plots for one range."""
    os.makedirs(output_dir, exist_ok=True)
    stats = results['stats']

    # Per-batch CSV
    per_batch_csv = os.path.join(output_dir, "per_batch_results.csv")
    with open(per_batch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_index", "batch_size", "accuracy", "error_rate",
            "BER", "psnr", "ssim", "lpips"
        ])
        for b in results['batch_results']:
            writer.writerow([
                b['batch_index'], b['batch_size'], b['accuracy'],
                b['error_rate'], b['BER'], b['psnr'], b['ssim'], b['lpips']
            ])

    # Summary txt
    summary_txt = os.path.join(output_dir, "results_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"FLAT RATIO {range_label} — TEST SUMMARY\n")
        f.write(f"Processed Images: {stats['processed_images']}\n\n")
        f.write(f"BER:      {stats['avg_ber']:.4f} ± {stats['std_ber']:.4f}\n")
        f.write(f"Accuracy: {stats['avg_accuracy']:.4f} ± {stats['std_accuracy']:.4f}\n")
        f.write(f"PSNR:     {stats['avg_psnr']:.2f} ± {stats['std_psnr']:.2f}\n")
        f.write(f"LPIPS:    {stats['avg_lpips']:.4f} ± {stats['std_lpips']:.4f}\n")
        f.write(f"SSIM:     {stats['avg_ssim']:.4f} ± {stats['std_ssim']:.4f}\n")

    # Metric distribution plots
    for metric_name, data in results['distributions'].items():
        if len(data) > 0:
            plot_metric_distribution(data, metric_name.upper(), output_dir)


# ------------------------------------------------
# Configuration
# ------------------------------------------------
# Initialize logging (creates test result files)
initialize_test_logging()

jpeg_quality = 50
jpeg_subsample = 2  # PIL: 2 = JPEG 4:2:0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create jpeg instance for testing
jpeg_sim_layer = JpegARWGAN420_v2(quality=jpeg_quality) 

print("Loading test settings...")
print("-----------------------------------")
print(f"experiment folder: {result_folder}")
match = re.search(r"(exp\d+)", result_folder)
print(f"match experimental name: {match}")
result_name = match.group(1) if match else "default_exp"

OUTPUT_DIR = (
    f"./Test_results/results_run({result_name})_epoch{model_epoch}"
    f"_JpegTest_420_jpeg{jpeg_quality}_ImageNet_balanced_test5000"
    f"_flat_color_presplit_jpeg{strength_factor}_with_YcbcrMSE_Attention_(baseline)_64bits_v2/"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# Dataset root — adjust this to your actual path
# ------------------------------------------------
# Expected structure under DATASET_ROOT:
#   test/                          ← all 5000 images
#   flat_ratio_0.00_0.20/test/
#   flat_ratio_0.20_0.40/test/
#   flat_ratio_0.40_0.60/test/
#   flat_ratio_0.60_0.80/test/
#   flat_ratio_0.80_1.00/test/
DATASET_ROOT = dataset_path   # from load_test_setting

# Discover all flat-ratio subdirectories automatically
FLAT_RATIO_DIRS = sorted([
    d for d in os.listdir(DATASET_ROOT)
    if d.startswith("flat_ratio_") and os.path.isdir(os.path.join(DATASET_ROOT, d))
])

# Build label → path mapping
# "all" uses the top-level test/ directory
subset_map = {"all": os.path.join(DATASET_ROOT, "test")}
for dir_name in FLAT_RATIO_DIRS:
    label = dir_name.replace("flat_ratio_", "").replace("_", "-")  # e.g. "0.00-0.20"
    subset_map[label] = os.path.join(DATASET_ROOT, dir_name, "test")

print(f"\nDiscovered subsets: {list(subset_map.keys())}")

# ------------------------------------------------
# Model
# ------------------------------------------------
noise_layers_alt = [
    f"JpegTest({jpeg_quality}, subsample={jpeg_subsample})"
]
# print(f"Using noise layer: {noise_layers_alt[0]} (JPEG 4:4:4)")

model = Model(
    H, W, message_length,
    noise_layers_alt,
    device,
    batch_size,
    lr,
    decoder_only_training=decoder_only_training
)

encoder_path = result_folder + f"models/Encoder_{model_epoch}.pth"
model.load_model_ed(encoder_path)

# ------------------------------------------------
# Main loop: test each subset
# ------------------------------------------------
print("\n" + "="*60)
print("TESTING PIPELINE — PRE-SPLIT FLAT COLOR RATIO DIRS")
print("="*60)

all_subset_results = {}

for label, data_dir in subset_map.items():
    if not os.path.isdir(data_dir):
        print(f"\n⚠️  Directory not found, skipping: {data_dir}")
        continue

    print(f"\n[Subset: {label}]  →  {data_dir}")

    dataset    = Dataset_loader(data_dir, H, W)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if len(dataset) == 0:
        print(f"   ⚠️  No images found, skipping.")
        continue

    results = run_test(
        model, dataloader, device, message_length,
        desc=f"Subset {label}",
        jpeg_sim_instance=jpeg_sim_layer
    )

    all_subset_results[label] = results

    # Save individual subset results
    subset_out_dir = os.path.join(
        OUTPUT_DIR,
        "all_images" if label == "all" else f"flat_ratio_{label.replace('-', '_')}"
    )
    save_range_results(results, label, subset_out_dir)

    stats = results['stats']
    print(f"   ✅  {stats['processed_images']} images | "
          f"BER={stats['avg_ber']:.4f} | "
          f"Acc={stats['avg_accuracy']:.4f} | "
          f"PSNR={stats['avg_psnr']:.2f} | "
          f"SSIM={stats['avg_ssim']:.4f} | "
          f"LPIPS={stats['avg_lpips']:.4f}")

# ------------------------------------------------
# Comparison CSV (flat-ratio subsets only, excl. "all")
# ------------------------------------------------
print("\nCreating comparison summary...")

ratio_results = {k: v for k, v in all_subset_results.items() if k != "all"}

comparison_csv = os.path.join(OUTPUT_DIR, "flat_ratio_comparison.csv")
with open(comparison_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Flat_Ratio_Range",
        "Processed_Images",
        "Avg_BER", "Std_BER",
        "Avg_Accuracy", "Std_Accuracy",
        "Avg_PSNR", "Std_PSNR",
        "Avg_LPIPS", "Std_LPIPS",
        "Avg_SSIM", "Std_SSIM",
    ])
    for label in sorted(ratio_results.keys()):
        s = ratio_results[label]['stats']
        writer.writerow([
            label,
            s['processed_images'],
            s['avg_ber'],      s['std_ber'],
            s['avg_accuracy'], s['std_accuracy'],
            s['avg_psnr'],     s['std_psnr'],
            s['avg_lpips'],    s['std_lpips'],
            s['avg_ssim'],     s['std_ssim'],
        ])

# Comparison bar-chart plot (flat-ratio subsets only)
if ratio_results:
    plot_flat_ratio_comparison(ratio_results, OUTPUT_DIR)

# ------------------------------------------------
# Final summary
# ------------------------------------------------
print("\n" + "="*60)
print("✅  ALL TESTING COMPLETED SUCCESSFULLY")
print("="*60)
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print(f"\n📊 Results structure:")
print(f"   ├── all_images/")
for label in sorted(ratio_results.keys()):
    tag = label.replace('-', '_')
    print(f"   ├── flat_ratio_{tag}/")
print(f"   ├── flat_ratio_comparison.csv")
print(f"   └── flat_ratio_comparison.png")
print()