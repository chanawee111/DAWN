# DAWN-A-Distortion-Aware-Watermarking-Network-with-Attention-Gating-under-JPEG-Compression
Image Steganography, Watermarking, Attention Mechanism , U-Net3+, JPEG Compression Robustness, Convolutional neural network

## Environment Setup

DAWN was developed and tested using Python 3.13 and PyTorch 2.6.0 with CUDA 12.4.

### 1. Clone the repository

```bash
git clone <YOUR_DAWN_REPOSITORY_URL>
cd DAWN-A-Distortion-Aware-Watermarking-Network-with-Attention-Gating-under-JPEG-Compression
```

### 2. Create the Conda environment

```bash
conda create -n DAWN python=3.13 -y
conda activate DAWN
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch

For CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 5. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

A successful CUDA installation should report:

```text
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
GPU: <Your NVIDIA GPU>
```

### Main Dependencies

* PyTorch 2.6.0
* Torchvision 0.21.0
* Kornia 0.8.0
* PyTorch-MSSSIM 1.0.0
* OpenCV 4.11.0
* scikit-image 0.25.2
* TensorBoard 2.19.0

The project focuses on image watermarking/steganography with distortion-aware attention gating, U-Net3+ based feature processing, wavelet-domain processing, and robustness against JPEG compression.

## Dataset Configuration

DAWN requires training, validation, and testing datasets. All should contain RGB images in JPEG or PNG format.

### Training Dataset

Place your training images in a flat directory with no subdirectories:

```
your_training_dataset/
├── image_001.jpeg
├── image_002.png
├── image_003.jpeg
└── ...
```

### Validation Dataset

Place your validation images in a flat directory with no subdirectories:

```
your_validation_dataset/
├── image_001.jpeg
├── image_002.png
├── image_003.jpeg
└── ...
```

### Configuration

Update `train_settings.json` with both dataset paths:
```json
{
  "dataset_path": "/path/to/your_training_dataset",
  "enable_validation": true,
  "validation_dataset_path": "/path/to/your_validation_dataset",
  "H": 128,
  "W": 128,
  "message_length": 30,
  "batch_size": 4,
  "epoch_number": 65
}
```

**Key Parameter:**
- `enable_validation` (true/false): Enable or disable validation during training. Set to `false` to skip validation and train faster.

### Testing Dataset

The test dataset must have a `test` subdirectory containing all test images:

```
your_test_dataset/
└── test/
    ├── image_001.jpeg
    ├── image_002.png
    ├── image_003.jpeg
    └── ...
```

Then update `test_settings.json` to point to the `test` subdirectory:
```json
{
  "dataset_path": "/path/to/your_test_dataset/test",
  "H": 128,
  "W": 128,
  "message_length": 30,
  "model_epoch": 65
}
```

### Dataset Requirements

- **Image Format:** JPEG or PNG
- **Image Size:** Any size (automatically resized to H×W during loading)
- **Color Space:** RGB (3-channel color images)
- **Training Dataset:** Typically 15,000-20,000+ images
- **Validation Dataset:** Typically 2,000-5,000 images
- **Test Dataset:** Typically 5,000+ images

## Training

### Quick Start

1. **Prepare your datasets** following the Dataset Configuration section above

2. **Update `train_settings.json`** with your dataset paths and desired parameters

3. **Start training:**
   ```bash
   python train.py
   ```

### Training Parameters

Adjust the following parameters in `train_settings.json`:

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `project_name` | Project name for results folder | String | `DAWN_JPEG420_30bits` |
| `decoder_only_training` | If true, only train decoder; if false, train encoder+decoder | Boolean | `false` |
| `epoch_number` | Total number of training epochs | Integer | `65` |
| `batch_size` | Number of images per batch | Integer | `4` |
| `discriminator_weight` | Weight for discriminator loss | Float | `0.001` |
| `dataset_path` | Path to training dataset | String | Required |
| `enable_validation` | Enable validation during training | Boolean | `true` |
| `validation_dataset_path` | Path to validation dataset | String | Required if `enable_validation` is true |
| `H` | Image height (images resized to this) | Integer | `128` |
| `W` | Image width (images resized to this) | Integer | `128` |
| `message_length` | Length of watermark message in bits | Integer | `30` |
| `lr` | Learning rate | Float | `1e-4` |
| `noise_layers` | Distortion layers for robustness training | List | `["Combined([JpegARWGAN420_v2(quality=50), Identity()])"]` |
| `save_images_number` | Number of sample images to save during training | Integer | `4` |
| `seed` | Random seed for reproducibility | Integer | `1751610149` |
| `train_continue` | Resume from checkpoint | Boolean | `false` |
| `train_continue_path` | Path to checkpoint folder | String | `""` |
| `train_continue_epoch` | Epoch to resume from | Integer | `65` |
| `train_continue_from_lastest_epoch` | Auto-detect latest epoch | Boolean | `false` |

### Training Example

```json
{
  "project_name": "DAWN_JPEG420_30bits",
  "decoder_only_training": false,
  "epoch_number": 65,
  "batch_size": 4,
  "discriminator_weight": 0.001,
  "dataset_path": "/path/to/training/dataset",
  "enable_validation": true,
  "validation_dataset_path": "/path/to/validation/dataset",
  "H": 128,
  "W": 128,
  "message_length": 30,
  "lr": 1e-4,
  "noise_layers": ["Combined([JpegARWGAN420_v2(quality=50), Identity()])"],
  "seed": 1751610149
}
```

### Training Output

Results are saved in `results/` folder with:
- `train_log.txt` - Training metrics per epoch
- `val_log.txt` - Validation metrics per epoch (if validation enabled)
- `train_params.txt` - Complete training configuration
- `models/` - Encoder and Discriminator checkpoints
- `tensorboard_logs/` - TensorBoard logs for visualization
