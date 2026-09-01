'''
Function to save images

By jzyustc, 2020/12/21

'''

import os
import numpy as np
import torch
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


def save_images(saved_all, epoch, folder, resize_to=None):
	original_images, watermarked_images, noised_images = saved_all

	images = original_images[:original_images.shape[0], :, :, :].cpu()
	watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

	# scale values to range [0, 1] from original range of [-1, 1]
	images = (images + 1) / 2
	watermarked_images = (watermarked_images + 1) / 2
	noised_images = (noised_images + 1) / 2

	# resize noised_images
	resize = nn.UpsamplingNearest2d(size=(images.shape[2], images.shape[3]))
	noised_images = resize(noised_images)

	if resize_to is not None:
		images = F.interpolate(images, size=resize_to)
		watermarked_images = F.interpolate(watermarked_images, size=resize_to)

	diff_images = (watermarked_images - images + 1) / 2

	# transform to rgb
	diff_images_linear = diff_images.clone()
	R = diff_images_linear[:, 0, :, :]
	G = diff_images_linear[:, 1, :, :]
	B = diff_images_linear[:, 2, :, :]
	diff_images_linear[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
	diff_images_linear[:, 1, :, :] = diff_images_linear[:, 0, :, :]
	diff_images_linear[:, 2, :, :] = diff_images_linear[:, 0, :, :]
	diff_images_linear = torch.abs(diff_images_linear * 2 - 1)

	# maximize diff in every image
	for id in range(diff_images_linear.shape[0]):
		diff_images_linear[id] = (diff_images_linear[id] - diff_images_linear[id].min()) / (
				diff_images_linear[id].max() - diff_images_linear[id].min())

	stacked_images = torch.cat(
		[images.unsqueeze(0), watermarked_images.unsqueeze(0), noised_images.unsqueeze(0), diff_images.unsqueeze(0),
		 diff_images_linear.unsqueeze(0)], dim=0)
	shape = stacked_images.shape
	stacked_images = stacked_images.permute(0, 3, 1, 4, 2).reshape(shape[3] * shape[0], shape[4] * shape[1], shape[2])
	stacked_images = stacked_images.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
	filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))

	saved_image = Image.fromarray(np.array(stacked_images, dtype=np.uint8)).convert("RGB")
	saved_image.save(filename)


def get_random_images(images, encoded_images, noised_images):
	selected_id = np.random.randint(1, images.shape[0]) if images.shape[0] > 1 else 1
	image = images.cpu()[selected_id - 1:selected_id, :, :, :]
	encoded_image = encoded_images.cpu()[selected_id - 1:selected_id, :, :, :]
	noised_image = noised_images.cpu()[selected_id - 1:selected_id, :, :, :]
	return [image, encoded_image, noised_image]


def concatenate_images(saved_all, images, encoded_images, noised_images):
	saved = get_random_images(images, encoded_images, noised_images)
	if saved_all[2].shape[2] != saved[2].shape[2]:
		return saved_all
	saved_all[0] = torch.cat((saved_all[0], saved[0]), 0)
	saved_all[1] = torch.cat((saved_all[1], saved[1]), 0)
	saved_all[2] = torch.cat((saved_all[2], saved[2]), 0)
	return saved_all

def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    # torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)
    torchvision.utils.save_image(
    stacked_images,
    filename,
    nrow=original_images.shape[0],
    normalize=False
    )
    
def save_images_separate(original_images, watermarked_images, epoch, folder, resize_to=None):
    # Move to CPU and scale from [-1, 1] to [0, 1]
    images = (original_images[:original_images.shape[0]].cpu() + 1) / 2
    watermarked_images = (watermarked_images[:watermarked_images.shape[0]].cpu() + 1) / 2

    # Resize if needed
    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    # Save original images (cover)
    cover_filename = os.path.join(folder, f'epoch-{epoch}-cover.png')
    torchvision.utils.save_image(
        images,
        cover_filename,
        nrow=original_images.shape[0],
        normalize=False
    )

    # Save watermarked (encoded) images
    encoded_filename = os.path.join(folder, f'epoch-{epoch}-encoded.png')
    torchvision.utils.save_image(
        watermarked_images,
        encoded_filename,
        nrow=original_images.shape[0],
        normalize=False
    )

def save_images_all(original_images, watermarked_images, noisy_images, epoch, folder, resize_to=None):
    # Move to CPU and scale values to [0,1]
    def process(imgs):
        imgs = imgs[:imgs.shape[0], :, :, :].cpu()
        imgs = (imgs + 1) / 2
        if resize_to is not None:
            imgs = F.interpolate(imgs, size=resize_to)
        return imgs

    images = process(original_images)
    watermarked_images = process(watermarked_images)
    noisy_images = process(noisy_images)

    # stack them vertically per sample: [cover, encoded, noisy]
    stacked_images = torch.cat([images, watermarked_images, noisy_images], dim=0)

    filename = os.path.join(folder, f'epoch-{epoch}.png')
    torchvision.utils.save_image(
        stacked_images,
        filename,
        nrow=original_images.shape[0],  # each row = one triplet
        normalize=False
    )

def save_images_separate_rs(original_images, watermarked_images, residual_images,
                         epoch, folder, resize_to=None):

    # ----------------------------------------
    # Prepare input images: scale [-1, 1] → [0, 1]
    # ----------------------------------------
    images = (original_images.cpu() + 1) / 2
    watermarked_images = (watermarked_images.cpu() + 1) / 2

    # Residual images come from the model directly (B,3,H,W)
    residual_images = residual_images

    # Resize if needed
    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)
        residual_images = F.interpolate(residual_images, size=resize_to)

    os.makedirs(folder, exist_ok=True)

    # ----------------------------------------
    # Save Cover & Encoded Images
    # ----------------------------------------
    cover_filename = os.path.join(folder, f'epoch-{epoch}-cover.png')
    torchvision.utils.save_image(images, cover_filename,
                                nrow=original_images.shape[0])

    encoded_filename = os.path.join(folder, f'epoch-{epoch}-encoded.png')
    torchvision.utils.save_image(watermarked_images, encoded_filename,
                                nrow=original_images.shape[0])

    # ----------------------------------------
    # Save Residual Grid (normalized)
    # ----------------------------------------
    # residual_norm = (residual_images - residual_images.min()) / \
    #                 (residual_images.max() - residual_images.min() + 1e-8)

    residual_rgb_filename = os.path.join(folder, f'epoch-{epoch}-residual-rgb.png')
    torchvision.utils.save_image(residual_images, residual_rgb_filename,
                                nrow=original_images.shape[0])

    # ----------------------------------------
    # Generate Residual Analysis per image
    # ----------------------------------------
    for idx in range(residual_images.shape[0]):
        img = residual_images[idx].permute(1, 2, 0).numpy()  # HWC format

        print(f"Residual Image {idx} size: {img.shape}")

        # Create combined analysis figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Residual Analysis - Image {idx}", fontsize=16, fontweight='bold')

        # 1. RGB Residual
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("RGB Residual")
        axes[0, 0].axis("off")

        # 2. Channels (R,G,B)
        channels = ["Red", "Green", "Blue"]
        colors = ["Reds", "Greens", "Blues"]
        for i in range(3):
            axes[0, 1 + (i // 2)].imshow(img[:, :, i], cmap=colors[i])
            axes[0, 1 + (i // 2)].set_title(f"{channels[i]} Channel")
            axes[0, 1 + (i // 2)].axis("off")

        # 3. Magnitude heatmap
        magnitude = np.sqrt(np.sum(img**2, axis=2))
        im = axes[1, 0].imshow(magnitude, cmap="hot")
        axes[1, 0].set_title("Embedding Magnitude")
        axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        # 4. Activity mask (top 25%)
        threshold = np.percentile(magnitude, 75)
        mask = magnitude > threshold
        axes[1, 1].imshow(mask, cmap="gray")
        axes[1, 1].set_title("High Activity Regions")
        axes[1, 1].axis("off")

        # 5. Histogram
        axes[1, 2].hist(img.flatten(), bins=50, alpha=0.7,
                        color="blue", edgecolor="black")
        axes[1, 2].set_title("Residual Value Distribution")
        axes[1, 2].set_xlabel("Pixel Value")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save instead of show
        fig_save_path = os.path.join(folder, f'epoch-{epoch}-residual-analysis-{idx}.png')
        plt.savefig(fig_save_path)
        plt.close()

        # Print statistics
        print(f"Residual Statistics:")
        print(f"  Mean: {np.mean(img):.6f}")
        print(f"  Std Dev: {np.std(img):.6f}")
        print(f"  Min: {np.min(img):.6f}, Max: {np.max(img):.6f}")
        print(f"  Magnitude Mean: {np.mean(magnitude):.6f}")
        print(f"  Active Regions: {np.sum(mask) / mask.size * 100:.2f}% of pixels\n")