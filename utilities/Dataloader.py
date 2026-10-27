
import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset


# Supported image formats; CSV and other non-image files are excluded.
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


class Dataset_loader(Dataset):

    def __init__(self, path, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W
        self.path = os.fspath(path)

        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.path}"
            )

        # Include only image files, not CSV files or subdirectories.
        self.list = sorted(
            filename
            for filename in os.listdir(self.path)
            if os.path.isfile(os.path.join(self.path, filename))
            and os.path.splitext(filename)[1].lower()
            in IMAGE_EXTENSIONS
        )

        if not self.list:
            raise RuntimeError(
                f"No supported image files found in: {self.path}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((int(self.H), int(self.W))),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

        print(f"Dataset directory: {self.path}")
        print(f"Image files found: {len(self.list)}")

    def transform_image(self, image):

        # Preserve the original image filtering rules.
        if (
            image.size[0] < self.W / 2
            and image.size[1] < self.H / 2
        ):
            return None

        if (
            image.size[0] < image.size[1] / 2
            or image.size[1] < image.size[0] / 2
        ):
            return None

        return self.transform(image)

    def __getitem__(self, index):

        # Try each image at most once to avoid an infinite loop.
        for offset in range(len(self.list)):
            current_index = (index + offset) % len(self.list)
            filename = self.list[current_index]
            image_path = os.path.join(self.path, filename)

            try:
                with Image.open(image_path) as img:
                    image = img.convert("RGB")

                image = self.transform_image(image)

                if image is not None:
                    return image

            except (OSError, ValueError, UnidentifiedImageError) as exc:
                print(f"Warning: skipping unreadable image: {filename}")
                print(f"Reason: {exc}")

        raise RuntimeError(
            f"No valid images could be loaded from: {self.path}. "
            "Check for corrupted images or images rejected by the "
            "size/aspect-ratio filters."
        )

    def __len__(self):
        return len(self.list)


class Dataset_loader_augmentation(Dataset):

    def __init__(self, path, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W
        self.path = os.fspath(path)

        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.path}"
            )

        # Include only image files.
        self.list = sorted(
            filename
            for filename in os.listdir(self.path)
            if os.path.isfile(os.path.join(self.path, filename))
            and os.path.splitext(filename)[1].lower()
            in IMAGE_EXTENSIONS
        )

        if not self.list:
            raise RuntimeError(
                f"No supported image files found in: {self.path}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((int(self.H), int(self.W))),

            # Original augmentations.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        print(f"Augmentation dataset directory: {self.path}")
        print(f"Image files found: {len(self.list)}")

    def transform_image(self, image):

        # Preserve the original image filtering rules.
        if (
            image.size[0] < self.W / 2
            and image.size[1] < self.H / 2
        ):
            return None

        if (
            image.size[0] < image.size[1] / 2
            or image.size[1] < image.size[0] / 2
        ):
            return None

        return self.transform(image)

    def __getitem__(self, index):

        # Try each image at most once to avoid an infinite loop.
        for offset in range(len(self.list)):
            current_index = (index + offset) % len(self.list)
            filename = self.list[current_index]
            image_path = os.path.join(self.path, filename)

            try:
                with Image.open(image_path) as img:
                    image = img.convert("RGB")

                image = self.transform_image(image)

                if image is not None:
                    return image

            except (OSError, ValueError, UnidentifiedImageError) as exc:
                print(f"Warning: skipping unreadable image: {filename}")
                print(f"Reason: {exc}")

        raise RuntimeError(
            f"No valid images could be loaded from: {self.path}. "
            "Check for corrupted images or images rejected by the "
            "size/aspect-ratio filters."
        )

    def __len__(self):
        return len(self.list)
