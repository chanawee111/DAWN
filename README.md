# DAWN-A-Distortion-Aware-Watermarking-Network-with-Attention-Gating-under-JPEG-Compression
Image Steganography, Watermarking, Attention Mechanism , U-Net3+, JPEG Compression Robustness, Convolutional neural network

# 1. Clone repository
git clone https://github.com/chanawee111/DAWN-A-Distortion-Aware-Watermarking-Network-with-Attention-Gating-under-JPEG-Compression.git
cd DAWN-A-Distortion-Aware-Watermarking-Network-with-Attention-Gating-under-JPEG-Compression

# 2. Create environment
conda create -n DAWN python=3.10 -y
conda activate DAWN

# 3. Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install CUDA-enabled PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Check CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"