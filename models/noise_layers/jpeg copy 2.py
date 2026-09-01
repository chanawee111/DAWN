import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

# -------------------- Color Space Conversions --------------------

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)

rgb_to_ycbcr_jpeg = rgb_to_ycbcr
ycbcr_to_rgb_jpeg = ycbcr_to_rgb

# -------------------- ENHANCED Chroma Subsampling --------------------

def downsampling_420(image):
    """
    4:2:0 chroma subsampling with enhanced boundary handling
    """
    y, cb, cr = torch.split(image, 1, dim=1)
    
    # ENHANCED: Use exact averaging for 2x2 blocks without any padding artifacts
    # This mimics real JPEG encoder behavior more closely
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2, padding=0)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2, padding=0)
    
    return y, cb, cr


def upsampling_420(y, cb, cr):
    """
    Upsample chroma with anti-aliasing for smoother edges
    """
    _, _, h, w = y.shape
    
    # ENHANCED: Use bicubic interpolation for smoother upsampling
    # This reduces visible artifacts at edges compared to bilinear
    cb = F.interpolate(cb, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
    cr = F.interpolate(cr, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
    
    return torch.cat([y, cb, cr], dim=1)


# -------------------- Block Operations --------------------

def image_to_patches(image):
    """
    Convert image to 8x8 patches (boundary-safe version)
    Input: [B, C, H, W]
    Output: [B, C, H*W//64, 8, 8]
    """
    k = 8
    b, c, h, w = image.shape
    
    # Ensure divisibility
    assert h % k == 0 and w % k == 0, f"Image dims {h}x{w} not divisible by {k}"
    
    # Unfold creates sliding windows
    patches = image.unfold(2, k, k).unfold(3, k, k)
    # Shape: [B, C, H//k, W//k, k, k]
    
    # Reshape to [B, C, num_patches, k, k]
    patches = patches.contiguous()
    patches = patches.view(b, c, -1, k, k)
    
    return patches


def patches_to_image(patches, height, width):
    """
    Reconstruct image from 8x8 patches (boundary-safe version)
    Input: [B, C, num_patches, 8, 8]
    Output: [B, C, H, W]
    """
    k = 8
    b, c, num_patches, _, _ = patches.shape
    
    # Reshape back to grid
    h_patches = height // k
    w_patches = width // k
    
    assert num_patches == h_patches * w_patches, \
        f"Patch count mismatch: {num_patches} != {h_patches}*{w_patches}"
    
    # Reshape to [B, C, H//k, W//k, k, k]
    patches = patches.view(b, c, h_patches, w_patches, k, k)
    
    # Permute to [B, C, H//k, k, W//k, k]
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    
    # Reshape to [B, C, H, W]
    image = patches.view(b, c, height, width)
    
    return image


# -------------------- DCT/IDCT --------------------

_dct_tensor = None
_idct_tensor = None

def get_dct_tensor(device):
    global _dct_tensor
    if _dct_tensor is None or _dct_tensor.device != device:
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        _dct_tensor = torch.from_numpy(tensor).to(device)
    return _dct_tensor


def get_dct_scale(device):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    scale = np.outer(alpha, alpha) * 0.25
    return torch.from_numpy(scale).to(device)


def dct_8x8(image):
    """Apply DCT to 8x8 image patches"""
    device = image.device
    tensor = get_dct_tensor(device)
    scale = get_dct_scale(device)
    
    image = image - 128.
    result = scale * torch.tensordot(image, tensor, dims=2)
    result = result.view(image.shape)
    return result


def get_idct_tensor(device):
    global _idct_tensor
    if _idct_tensor is None or _idct_tensor.device != device:
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        _idct_tensor = torch.from_numpy(tensor).to(device)
    return _idct_tensor


def get_idct_alpha(device):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    alpha = np.outer(alpha, alpha)
    return torch.from_numpy(alpha).to(device)


def idct_8x8(image):
    """Apply Inverse DCT to 8x8 patches"""
    device = image.device
    tensor = get_idct_tensor(device)
    alpha = get_idct_alpha(device)
    
    image = image * alpha
    result = 0.25 * torch.tensordot(image, tensor, dims=2)
    result = result.view(image.shape) + 128
    return result


# -------------------- Quantization Tables --------------------

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], 
     [12, 12, 14, 19, 26, 58, 60, 55], 
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], 
     [18, 22, 37, 56, 68, 109, 103, 77], 
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], 
     [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T


def y_quantize(image, rounding, factor=1):
    table = torch.from_numpy(y_table).to(image.device)
    image = image / (table * factor)
    image = rounding(image)
    return image


def c_quantize(image, rounding, factor=1):
    table = torch.from_numpy(c_table).to(image.device)
    image = image / (table * factor)
    image = rounding(image)
    return image


def y_dequantize(image, factor=1):
    table = torch.from_numpy(y_table).to(image.device)
    return image * (table * factor)


def c_dequantize(image, factor=1):
    table = torch.from_numpy(c_table).to(image.device)
    return image * (table * factor)


# -------------------- Differentiable Rounding --------------------

def diff_round(x):
    """Differentiable approximation of rounding"""
    return torch.round(x) + (x - torch.round(x)) ** 3


def diff_round_straight_through(x):
    """
    Straight-through estimator for rounding
    Forward: actual rounding
    Backward: pass gradient through as identity
    This prevents the model from exploiting rounding gradients at edges
    """
    return x + (torch.round(x) - x).detach()


def stochastic_round(x):
    """
    Stochastic rounding: randomly round up or down based on fractional part
    This breaks deterministic patterns the model tries to learn
    """
    frac = x - torch.floor(x)
    # Random decision based on fractional part
    rand_mask = torch.rand_like(x) < frac
    rounded = torch.where(rand_mask, torch.ceil(x), torch.floor(x))
    # Straight-through for gradients
    return x + (rounded - x).detach()


# -------------------- Quality Factor --------------------

def quality_to_factor(quality, min_factor=0.1):
    """Convert JPEG quality (1-100) to quantization factor"""
    quality = max(1, min(100, quality))
    if quality < 50:
        factor = 50. / quality
    else:
        factor = 2. - quality * 0.02
    return max(factor, min_factor)


def random_quality_to_factor(min_quality=70, max_quality=95, min_factor=0.1):
    """
    Randomly sample JPEG quality and convert to quantization factor
    
    Args:
        min_quality: Minimum quality value (default: 70)
        max_quality: Maximum quality value (default: 95)
        min_factor: Minimum quantization factor
    
    Returns:
        factor: Random quantization factor
    """
    quality = torch.randint(min_quality, max_quality + 1, (1,)).item()
    return quality_to_factor(quality, min_factor)


def weighted_random_quality_to_factor(min_quality=50, max_quality=95, 
                                      target_quality=50, weight=0.5, 
                                      min_factor=0.1):
    """
    Sample quality with higher probability at target_quality
    
    Args:
        min_quality: Minimum quality value (default: 50)
        max_quality: Maximum quality value (default: 95)
        target_quality: Quality value to sample more often (default: 50)
        weight: Probability of sampling target_quality (0.0-1.0, default: 0.5)
        min_factor: Minimum quantization factor
    
    Returns:
        factor: Random quantization factor
    
    Example:
        # 50% chance of quality=50, 50% chance of quality in [50, 95]
        factor = weighted_random_quality_to_factor(50, 95, target_quality=50, weight=0.5)
    """
    if torch.rand(1).item() < weight:
        quality = target_quality
    else:
        quality = torch.randint(min_quality, max_quality + 1, (1,)).item()
    
    return quality_to_factor(quality, min_factor)


# -------------------- Main JPEG Compression --------------------

def jpeg_compress_decompress(image, downsample_c=False, rounding=diff_round, factor=1., 
                            add_noise=False, noise_std=0.5, random_shift=False):
    """
    Differentiable JPEG compression/decompression
    ENHANCED: Multiple fixes for edge artifacts in steganography
    
    Args:
        add_noise: If True, add small random noise to prevent overfitting
        noise_std: Standard deviation of noise (default: 0.5/255 scale)
        random_shift: If True, randomly shift image before processing (CRITICAL for edge artifacts)
    """
    device = image.device
    
    # Convert from [-1,1] to [0,255]
    image = (image + 1) / 2 * 255
    b, c, orig_h, orig_w = image.shape
    
    # Store original dimensions for final output
    final_h, final_w = orig_h, orig_w

    # CRITICAL NEW FIX: Random spatial shift to break edge pattern learning
    shift_h, shift_w = 0, 0
    if random_shift and image.requires_grad:
        # Randomly shift by 0-7 pixels (less than one 8x8 block)
        shift_h = torch.randint(0, 8, (1,)).item()
        shift_w = torch.randint(0, 8, (1,)).item()
        if shift_h > 0 or shift_w > 0:
            image = F.pad(image, [shift_w, 0, shift_h, 0], mode='replicate')
            # Update working dimensions
            orig_h = orig_h + shift_h
            orig_w = orig_w + shift_w

    # Padding to multiple of 16
    if orig_h % 16 == 0 and orig_w % 16 == 0:
        h, w = orig_h, orig_w
        padded = False
        pad_top, pad_left = 0, 0
    else:
        h = ((orig_h + 15) // 16) * 16
        w = ((orig_w + 15) // 16) * 16
        vpad = h - orig_h
        wpad = w - orig_w
        pad_top = vpad // 2
        bottom = vpad - pad_top
        pad_left = wpad // 2
        right = wpad - pad_left
        image = F.pad(image, [pad_left, right, pad_top, bottom], mode='replicate')
        padded = True

    # --- COMPRESSION ---
    image = rgb_to_ycbcr_jpeg(image)
    
    if downsample_c:
        y, cb, cr = downsampling_420(image)
        components = {
            'y': (y, h, w),
            'cb': (cb, h // 2, w // 2),
            'cr': (cr, h // 2, w // 2)
        }
    else:
        y, cb, cr = torch.split(image, 1, dim=1)
        components = {
            'y': (y, h, w),
            'cb': (cb, h, w),
            'cr': (cr, h, w)
        }
    
    # Process each component
    processed = {}
    for key, (comp, comp_h, comp_w) in components.items():
        patches = image_to_patches(comp)
        b, c, num_p, k, k = patches.shape
        
        patches = patches.view(b * c, num_p, k, k)
        
        # DCT
        patches = dct_8x8(patches)
        
        # Quantize
        if key in ('cb', 'cr'):
            patches = c_quantize(patches, rounding, factor)
        else:
            patches = y_quantize(patches, rounding, factor)
        
        processed[key] = (patches, comp_h, comp_w, c)

    # --- DECOMPRESSION ---
    reconstructed = {}
    for key, (patches, comp_h, comp_w, c) in processed.items():
        # Dequantize
        if key in ('cb', 'cr'):
            patches = c_dequantize(patches, factor)
        else:
            patches = y_dequantize(patches, factor)
        
        # IDCT
        patches = idct_8x8(patches)
        
        # Reshape back
        b_times_c = patches.shape[0]
        b = b_times_c // c
        patches = patches.view(b, c, -1, 8, 8)
        
        # Reconstruct image
        comp = patches_to_image(patches, comp_h, comp_w)
        
        reconstructed[key] = comp

    y = reconstructed['y']
    cb = reconstructed['cb']
    cr = reconstructed['cr']
    
    # Upsample chroma if needed
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.cat([y, cb, cr], dim=1)
    
    # Convert back to RGB
    image = ycbcr_to_rgb_jpeg(image)

    # Crop back to working size (remove 16-alignment padding)
    if padded:
        image = image[:, :, pad_top:pad_top+orig_h, pad_left:pad_left+orig_w]
    
    # Crop back to original size (remove random shift)
    if random_shift and (shift_h > 0 or shift_w > 0):
        image = image[:, :, shift_h:shift_h+final_h, shift_w:shift_w+final_w]
    
    # ANTI-OVERFITTING: Add small random noise during training
    # This prevents the model from learning exact JPEG artifacts
    if add_noise and image.requires_grad:
        noise = torch.randn_like(image) * noise_std
        image = image + noise

    # Clamp and convert back to [-1, 1]
    image = torch.clamp(image, 0, 255)
    image = (image / 255.0) * 2.0 - 1.0

    return image


# -------------------- PyTorch Module Wrappers --------------------

class JpegARWGAN420_V2(nn.Module):
    """JPEG compression layer with 4:2:0 subsampling"""
    def __init__(self, factor=None, min_quality=70, max_quality=95, 
                 random_quality=False, weighted_quality=False, 
                 target_quality=50, target_weight=0.5,
                 rounding_mode='straight_through',
                 add_noise=False, noise_std=0.5,
                 random_shift=False):
        """
        Args:
            factor: Fixed quality factor (if None and random_quality=False, uses quality=75)
            min_quality: Minimum quality for random sampling (default: 70)
            max_quality: Maximum quality for random sampling (default: 95)
            random_quality: If True, randomly sample quality uniformly
            weighted_quality: If True, sample quality with bias toward target_quality
            target_quality: Target quality for weighted sampling (default: 50)
            target_weight: Probability of using target_quality (default: 0.5)
            rounding_mode: 'diff' | 'straight_through' | 'stochastic'
                - 'diff': Differentiable cubic approximation
                - 'straight_through': Identity gradients (recommended)
                - 'stochastic': Random rounding (BEST for edge artifacts)
            add_noise: If True, add random noise to prevent overfitting (default: False)
            noise_std: Standard deviation of noise in [0-255] scale (default: 0.5)
            random_shift: If True, randomly shift 8x8 blocks (CRITICAL for edges) (default: False)
        
        Usage Examples:
            # MOST AGGRESSIVE (try this for persistent edge artifacts):
            JpegARWGAN420(
                weighted_quality=True, target_quality=50, target_weight=0.5,
                min_quality=50, max_quality=95,
                rounding_mode='stochastic',  # ⭐ NEW: Breaks patterns
                add_noise=True, noise_std=2.0,
                random_shift=True  # ⭐ NEW: Breaks edge learning
            )
        """
        super(JpegARWGAN420_V2, self).__init__()
        self.random_quality = random_quality
        self.weighted_quality = weighted_quality
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.target_quality = target_quality
        self.target_weight = target_weight
        self.rounding_mode = rounding_mode
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_shift = random_shift
        
        if factor is not None:
            self.factor = factor
        else:
            self.factor = quality_to_factor(75)

    def forward(self, noise_and_cover):
        encoded_image = noise_and_cover[0]
        
        # Determine rounding function
        if self.rounding_mode == 'stochastic':
            rounding = stochastic_round
        elif self.rounding_mode == 'straight_through':
            rounding = diff_round_straight_through
        else:  # 'diff'
            rounding = diff_round
        
        # Determine quality factor
        if self.weighted_quality:
            factor = weighted_random_quality_to_factor(
                self.min_quality, 
                self.max_quality,
                self.target_quality,
                self.target_weight
            )
        elif self.random_quality:
            factor = random_quality_to_factor(self.min_quality, self.max_quality)
        else:
            factor = self.factor
        
        jpeg_image = jpeg_compress_decompress(
            encoded_image, 
            downsample_c=True,
            rounding=rounding,
            factor=factor,
            add_noise=self.add_noise,
            noise_std=self.noise_std,
            random_shift=self.random_shift
        )
        noise_and_cover[0] = jpeg_image
        return noise_and_cover[0]