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
* DTCWT 0.14.0
* PyTorch Wavelets 1.3.0
* Kornia 0.8.0
* PyTorch-MSSSIM 1.0.0
* TIMM 1.0.28
* OpenCV 4.11.0
* scikit-image 0.25.2
* TensorBoard 2.19.0

The project focuses on image watermarking/steganography with distortion-aware attention gating, U-Net3+ based feature processing, wavelet-domain processing, and robustness against JPEG compression.
