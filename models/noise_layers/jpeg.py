import os
import itertools
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import random, string

#JpegTest class to use differentiable torch operations instead of PIL. This involves implementing the JPEG compression pipeline (DCT, quantization, dequantization, IDCT) directly in PyTorch.

class ColorSpaceConverter(nn.Module):
    """Handles RGB <-> YCbCr color space conversions."""
    
    def __init__(self):
        super(ColorSpaceConverter, self).__init__()
    
    def rgb_to_ycbcr(self, rgb):
        """Convert RGB to YCbCr color space.
        
        Args:
            rgb: (B, 3, H, W) tensor in [0, 1] range
        
        Returns:
            ycbcr: (B, 3, H, W) tensor
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        
        return torch.cat([y, cb, cr], dim=1)
    
    def ycbcr_to_rgb(self, ycbcr):
        """Convert YCbCr to RGB color space.
        
        Args:
            ycbcr: (B, 3, H, W) tensor
        
        Returns:
            rgb: (B, 3, H, W) tensor in [0, 1] range
        """
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        
        cb = cb - 0.5
        cr = cr - 0.5
        
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        
        return torch.cat([r, g, b], dim=1)


class ChromaSubsampler(nn.Module):
    """Handles chroma subsampling and upsampling."""
    
    def __init__(self, subsample=2):
        """
        Args:
            subsample: 0='4:4:4', 1='4:2:2', 2='4:2:0'
        """
        super(ChromaSubsampler, self).__init__()
        self.subsample = subsample
    
    def downsample(self, cb, cr):
        """Apply chroma subsampling.
        
        Args:
            cb, cr: (B, 1, H, W) tensors
        
        Returns:
            Downsampled cb, cr tensors
        """
        if self.subsample == 0:  # 4:4:4 - no subsampling
            return cb, cr
        elif self.subsample == 1:  # 4:2:2 - horizontal subsampling
            cb = F.avg_pool2d(cb, kernel_size=(1, 2), stride=(1, 2))
            cr = F.avg_pool2d(cr, kernel_size=(1, 2), stride=(1, 2))
        elif self.subsample == 2:  # 4:2:0 - both directions
            cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
            cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
        
        return cb, cr
    
    def upsample(self, cb, cr, target_size):
        """Upsample chroma channels back to original size.
        
        Args:
            cb, cr: Downsampled tensors
            target_size: (H, W) tuple for target dimensions
        
        Returns:
            Upsampled cb, cr tensors
        """
        if self.subsample == 0:
            return cb, cr
        
        cb = F.interpolate(cb, size=target_size, mode='bilinear', align_corners=False)
        cr = F.interpolate(cr, size=target_size, mode='bilinear', align_corners=False)
        
        return cb, cr


class QuantizationTable(nn.Module):
    """Generates and stores JPEG quantization tables."""
    
    def __init__(self, quality=50):
        """
        Args:
            quality: JPEG quality factor (1-100)
        """
        super(QuantizationTable, self).__init__()
        self.quality = quality
        
        self.register_buffer('y_table', self._create_table('luma', quality))
        self.register_buffer('c_table', self._create_table('chroma', quality))
    
    def _create_table(self, table_type, quality):
        """Generate JPEG quantization table based on quality factor."""
        # Standard JPEG quantization tables
        if table_type == 'luma':
            base_table = torch.tensor([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ], dtype=torch.float32)
        else:  # chroma
            base_table = torch.tensor([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ], dtype=torch.float32)
        
        # Scale based on quality
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        
        table = torch.floor((base_table * scale + 50) / 100)
        table = torch.clamp(table, 1, 255)
        
        return table


class DCTTransform(nn.Module):
    """Handles DCT and IDCT transformations."""
    
    def __init__(self):
        super(DCTTransform, self).__init__()
        self.register_buffer('dct_matrix', self._create_dct_matrix())
    
    def _create_dct_matrix(self):
        """Create 8x8 DCT transformation matrix."""
        dct = torch.zeros(8, 8)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    dct[i, j] = 1.0 / np.sqrt(8)
                else:
                    dct[i, j] = np.sqrt(2.0 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
        return dct
    
    def forward_dct(self, blocks):
        """Apply DCT transform: D * block * D^T
        
        Args:
            blocks: (B, C, N, 8, 8) tensor
        
        Returns:
            DCT coefficients
        """
        dct_matrix = self.dct_matrix.to(blocks.device)
        dct_blocks = torch.matmul(dct_matrix, blocks)
        dct_blocks = torch.matmul(dct_blocks, dct_matrix.t())
        return dct_blocks
    
    def inverse_dct(self, blocks):
        """Apply IDCT transform: D^T * block * D
        
        Args:
            blocks: (B, C, N, 8, 8) tensor of DCT coefficients
        
        Returns:
            Spatial domain blocks
        """
        dct_matrix = self.dct_matrix.to(blocks.device)
        idct_blocks = torch.matmul(dct_matrix.t(), blocks)
        idct_blocks = torch.matmul(idct_blocks, dct_matrix)
        return idct_blocks


class JPEGEncoder(nn.Module):
    """JPEG encoding: DCT -> Quantization -> Huffman-like zeroing."""
    
    def __init__(self, quality=50, use_diff_round=True):
        super(JPEGEncoder, self).__init__()
        self.dct_transform = DCTTransform()
        self.q_tables = QuantizationTable(quality)
        self.huffman_threshold = 1.0 # to adjust the aggressiveness, you can modify the 1.0: Moderate, 0.5: Less aggressive, 2.0: More aggressive (zeros out more coefficients)
        self.use_diff_round = use_diff_round

    def diff_round(self, x):
        """Differentiable rounding for gradient flow."""
        return torch.round(x) + (x - torch.round(x)) ** 3

    def encode_channel(self, channel, q_table):
        """Encode a single channel.
        
        Args:
            channel: (B, 1, H, W) tensor in [0, 1] range
            q_table: (8, 8) quantization table
        
        Returns:
            Quantized DCT coefficients and metadata for decoding
        """
        B, C, H, W = channel.shape
        
        # Pad to make dimensions divisible by 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        channel_padded = F.pad(channel, (0, pad_w, 0, pad_h), mode='reflect')
        
        _, _, H_pad, W_pad = channel_padded.shape
        
        # Reshape into 8x8 blocks
        blocks = channel_padded.unfold(2, 8, 8).unfold(3, 8, 8)
        blocks = blocks.contiguous().view(B, C, -1, 8, 8)
        
        # Center around 0
        blocks = blocks - 0.5
        
        # Apply DCT
        dct_blocks = self.dct_transform.forward_dct(blocks)
        
        # Quantize
        q_table_expanded = q_table.to(blocks.device).view(1, 1, 1, 8, 8)
        if self.use_diff_round:
            quantized = self.diff_round(dct_blocks / q_table_expanded)
        else:
            quantized = torch.round(dct_blocks / q_table_expanded)
        
        # Huffman-like zeroing
        quantized = quantized * (torch.abs(quantized) >= self.huffman_threshold).float()
        
        metadata = {
            'original_size': (H, W),
            'padded_size': (H_pad, W_pad),
            'num_blocks': (H_pad // 8, W_pad // 8)
        }
        
        return quantized, metadata
    
    def forward(self, y, cb, cr):
        """Encode Y, Cb, Cr channels.
        
        Returns:
            Encoded coefficients and metadata
        """
        y_encoded, y_meta = self.encode_channel(y, self.q_tables.y_table)
        cb_encoded, cb_meta = self.encode_channel(cb, self.q_tables.c_table)
        cr_encoded, cr_meta = self.encode_channel(cr, self.q_tables.c_table)
        
        return {
            'y': (y_encoded, y_meta),
            'cb': (cb_encoded, cb_meta),
            'cr': (cr_encoded, cr_meta)
        }


class JPEGDecoder(nn.Module):
    """JPEG decoding: Dequantization -> IDCT."""
    
    def __init__(self, quality=50):
        super(JPEGDecoder, self).__init__()
        self.dct_transform = DCTTransform()
        self.q_tables = QuantizationTable(quality)
    
    def decode_channel(self, quantized, metadata, q_table):
        """Decode a single channel.
        
        Args:
            quantized: Quantized DCT coefficients
            metadata: Dictionary with decoding information
            q_table: (8, 8) quantization table
        
        Returns:
            Reconstructed channel (B, 1, H, W)
        """
        B, C, _, _, _ = quantized.shape
        H, W = metadata['original_size']
        H_pad, W_pad = metadata['padded_size']
        num_blocks_h, num_blocks_w = metadata['num_blocks']
        
        # Dequantize
        q_table_expanded = q_table.to(quantized.device).view(1, 1, 1, 8, 8)
        dequantized = quantized * q_table_expanded
        
        # Apply IDCT
        idct_blocks = self.dct_transform.inverse_dct(dequantized)
        
        # Add back 0.5
        idct_blocks = idct_blocks + 0.5
        
        # Reshape back to image
        idct_blocks = idct_blocks.view(B, C, num_blocks_h, num_blocks_w, 8, 8)
        idct_blocks = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        result = idct_blocks.view(B, C, H_pad, W_pad)
        
        # Remove padding
        if H_pad != H or W_pad != W:
            result = result[:, :, :H, :W]
        
        return result
    
    def forward(self, encoded_data):
        """Decode Y, Cb, Cr channels.
        
        Args:
            encoded_data: Dictionary from encoder
        
        Returns:
            Decoded y, cb, cr channels
        """
        y_quantized, y_meta = encoded_data['y']
        cb_quantized, cb_meta = encoded_data['cb']
        cr_quantized, cr_meta = encoded_data['cr']
        
        y = self.decode_channel(y_quantized, y_meta, self.q_tables.y_table)
        cb = self.decode_channel(cb_quantized, cb_meta, self.q_tables.c_table)
        cr = self.decode_channel(cr_quantized, cr_meta, self.q_tables.c_table)
        
        return y, cb, cr


class DifferentiableJpeg(nn.Module):
    """Complete differentiable JPEG compression pipeline."""
    
    def __init__(self, Q=50, subsample=2):
        """
        Args:
            Q: JPEG quality factor (1-100)
            subsample: Chroma subsampling (0='4:4:4', 1='4:2:2', 2='4:2:0')
        """
        super(DifferentiableJpeg, self).__init__()
        
        self.color_converter = ColorSpaceConverter()
        self.chroma_subsampler = ChromaSubsampler(subsample)
        self.encoder = JPEGEncoder(Q)
        self.decoder = JPEGDecoder(Q)
    
    def forward(self, image_and_cover):
        """
        Apply differentiable JPEG compression.
        
        Args:
            image_and_cover: Tuple of (image, cover_image)
                image shape: (B, 3, H, W), values in [-1, 1]
        
        Returns:
            Compressed image with same shape and range as input
        """
        image, cover_image = image_and_cover
        
        # Convert from [-1, 1] to [0, 1]
        image_01 = (image.clamp(-1, 1) + 1) / 2
        
        # RGB to YCbCr
        ycbcr = self.color_converter.rgb_to_ycbcr(image_01)
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        
        # Store original size
        _, _, H, W = y.shape
        
        # Chroma subsampling
        cb_sub, cr_sub = self.chroma_subsampler.downsample(cb, cr)
        
        # Encode
        encoded_data = self.encoder(y, cb_sub, cr_sub)
        
        # Decode
        y_decoded, cb_decoded, cr_decoded = self.decoder(encoded_data)
        
        # Chroma upsampling
        cb_up, cr_up = self.chroma_subsampler.upsample(cb_decoded, cr_decoded, (H, W))
        
        # Combine channels
        ycbcr_decoded = torch.cat([y_decoded, cb_up, cr_up], dim=1)
        
        # YCbCr to RGB
        rgb_decoded = self.color_converter.ycbcr_to_rgb(ycbcr_decoded)
        
        # Clamp and convert back to [-1, 1]
        rgb_decoded = torch.clamp(rgb_decoded, 0, 1)
        result = rgb_decoded * 2 - 1
        
        return result

# class DifferentiableJpeg(nn.Module):
#     def __init__(self, Q=50, subsample=2):
#         """
#         Differentiable JPEG compression using PyTorch operations.
        
#         Args:
#             Q: JPEG quality factor (1-100)
#             subsample: Chroma subsampling (0='4:4:4', 1='4:2:2', 2='4:2:0')
#         """
#         super(DifferentiableJpeg, self).__init__()
#         self.Q = Q
#         self.subsample = subsample
        
#         # Register quantization tables as buffers
#         self.register_buffer('y_table', self._get_quantization_table(Q, 'luma'))
#         self.register_buffer('c_table', self._get_quantization_table(Q, 'chroma'))
        
#         # Register DCT matrix as buffer
#         self.register_buffer('dct_matrix', self._get_dct_matrix())
        
#     def _get_dct_matrix(self):
#         """Create 8x8 DCT transformation matrix."""
#         dct = torch.zeros(8, 8)
#         for i in range(8):
#             for j in range(8):
#                 if i == 0:
#                     dct[i, j] = 1.0 / np.sqrt(8)
#                 else:
#                     dct[i, j] = np.sqrt(2.0 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
#         return dct
    
#     def _get_quantization_table(self, quality, table_type='luma'):
#         """Generate JPEG quantization table based on quality factor."""
#         # Standard JPEG quantization tables
#         if table_type == 'luma':
#             base_table = torch.tensor([
#                 [16, 11, 10, 16, 24, 40, 51, 61],
#                 [12, 12, 14, 19, 26, 58, 60, 55],
#                 [14, 13, 16, 24, 40, 57, 69, 56],
#                 [14, 17, 22, 29, 51, 87, 80, 62],
#                 [18, 22, 37, 56, 68, 109, 103, 77],
#                 [24, 35, 55, 64, 81, 104, 113, 92],
#                 [49, 64, 78, 87, 103, 121, 120, 101],
#                 [72, 92, 95, 98, 112, 100, 103, 99]
#             ], dtype=torch.float32)
#         else:  # chroma
#             base_table = torch.tensor([
#                 [17, 18, 24, 47, 99, 99, 99, 99],
#                 [18, 21, 26, 66, 99, 99, 99, 99],
#                 [24, 26, 56, 99, 99, 99, 99, 99],
#                 [47, 66, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99]
#             ], dtype=torch.float32)
        
#         # Scale based on quality
#         if quality < 50:
#             scale = 5000 / quality
#         else:
#             scale = 200 - 2 * quality
        
#         table = torch.floor((base_table * scale + 50) / 100)
#         table = torch.clamp(table, 1, 255)
        
#         return table
    
#     def _rgb_to_ycbcr(self, rgb):
#         """Convert RGB to YCbCr color space."""
#         # rgb shape: (B, 3, H, W), values in [0, 1]
#         r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
#         y = 0.299 * r + 0.587 * g + 0.114 * b
#         cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
#         cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        
#         return torch.cat([y, cb, cr], dim=1)
    
#     def _ycbcr_to_rgb(self, ycbcr):
#         """Convert YCbCr to RGB color space."""
#         # ycbcr shape: (B, 3, H, W)
#         y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        
#         cb = cb - 0.5
#         cr = cr - 0.5
        
#         r = y + 1.402 * cr
#         g = y - 0.344136 * cb - 0.714136 * cr
#         b = y + 1.772 * cb
        
#         return torch.cat([r, g, b], dim=1)
    
#     def _chroma_subsample(self, cb, cr):
#         """Apply chroma subsampling."""
#         if self.subsample == 0:  # 4:4:4 - no subsampling
#             return cb, cr
#         elif self.subsample == 1:  # 4:2:2 - horizontal subsampling
#             cb = F.avg_pool2d(cb, kernel_size=(1, 2), stride=(1, 2))
#             cr = F.avg_pool2d(cr, kernel_size=(1, 2), stride=(1, 2))
#         elif self.subsample == 2:  # 4:2:0 - both directions
#             cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
#             cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
        
#         return cb, cr
    
#     def _chroma_upsample(self, cb, cr, target_size):
#         """Upsample chroma channels back to original size."""
#         if self.subsample == 0:
#             return cb, cr
        
#         # Use bilinear interpolation for upsampling
#         cb = F.interpolate(cb, size=target_size, mode='bilinear', align_corners=False)
#         cr = F.interpolate(cr, size=target_size, mode='bilinear', align_corners=False)
        
#         return cb, cr
    
#     def _block_dct(self, channel, q_table):
#         """Apply DCT, quantization, dequantization, and IDCT on 8x8 blocks."""
#         B, C, H, W = channel.shape
        
#         # Pad to make dimensions divisible by 8
#         pad_h = (8 - H % 8) % 8
#         pad_w = (8 - W % 8) % 8
#         channel = F.pad(channel, (0, pad_w, 0, pad_h), mode='reflect')
        
#         _, _, H_pad, W_pad = channel.shape
        
#         # Reshape into 8x8 blocks: (B, C, H/8, 8, W/8, 8)
#         blocks = channel.unfold(2, 8, 8).unfold(3, 8, 8)
#         blocks = blocks.contiguous().view(B, C, -1, 8, 8)
        
#         # Center values around 0 (subtract 128 for 8-bit range [0, 255])
#         blocks = blocks - 0.5
        
#         # Apply DCT: D * block * D^T
#         dct_matrix = self.dct_matrix.to(blocks.device)
#         dct_blocks = torch.matmul(dct_matrix, blocks)
#         dct_blocks = torch.matmul(dct_blocks, dct_matrix.t())
        
#         # Quantize
#         q_table = q_table.to(blocks.device).view(1, 1, 1, 8, 8)
#         quantized = torch.round(dct_blocks / q_table)
        
#         # Huffman-like zeroing: zero out small coefficients that would be discarded
#         # In real JPEG, coefficients close to zero are often run-length encoded as zeros
#         # We simulate this by zeroing coefficients with absolute value < threshold
#         threshold = 1.0  # Coefficients smaller than this become zero
#         quantized = quantized * (torch.abs(quantized) >= threshold).float()
        
#         # Dequantize
#         dequantized = quantized * q_table
        
#         # Apply IDCT: D^T * block * D
#         idct_blocks = torch.matmul(dct_matrix.t(), dequantized)
#         idct_blocks = torch.matmul(idct_blocks, dct_matrix)
        
#         # Add back 128
#         idct_blocks = idct_blocks + 0.5
        
#         # Reshape back to image
#         num_blocks_h = H_pad // 8
#         num_blocks_w = W_pad // 8
#         idct_blocks = idct_blocks.view(B, C, num_blocks_h, num_blocks_w, 8, 8)
#         idct_blocks = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
#         result = idct_blocks.view(B, C, H_pad, W_pad)
        
#         # Remove padding
#         if pad_h > 0 or pad_w > 0:
#             result = result[:, :, :H, :W]
        
#         return result
    
#     def forward(self, image_and_cover):
#         """
#         Apply differentiable JPEG compression.
        
#         Args:
#             image_and_cover: Tuple of (image, cover_image)
#                 image shape: (B, 3, H, W), values in [-1, 1]
        
#         Returns:
#             Compressed image with same shape and range as input
#         """
#         image, cover_image = image_and_cover
        
#         # Convert from [-1, 1] to [0, 1]
#         image_01 = (image.clamp(-1, 1) + 1) / 2
        
#         # RGB to YCbCr
#         ycbcr = self._rgb_to_ycbcr(image_01)
#         y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        
#         # Store original size for upsampling
#         _, _, H, W = y.shape
        
#         # Chroma subsampling
#         cb_sub, cr_sub = self._chroma_subsample(cb, cr)
        
#         # Apply DCT + Quantization + IDCT
#         y_compressed = self._block_dct(y, self.y_table)
#         cb_compressed = self._block_dct(cb_sub, self.c_table)
#         cr_compressed = self._block_dct(cr_sub, self.c_table)
        
#         # Chroma upsampling
#         cb_up, cr_up = self._chroma_upsample(cb_compressed, cr_compressed, (H, W))
        
#         # Combine channels
#         ycbcr_compressed = torch.cat([y_compressed, cb_up, cr_up], dim=1)
        
#         # YCbCr to RGB
#         rgb_compressed = self._ycbcr_to_rgb(ycbcr_compressed)
        
#         # Clamp to valid range and convert back to [-1, 1]
#         rgb_compressed = torch.clamp(rgb_compressed, 0, 1)
#         result = rgb_compressed * 2 - 1
        
#         return result


# class DifferentiableJpeg(nn.Module):
#     def __init__(self, Q=50, subsample=2):
#         """
#         Differentiable JPEG compression using PyTorch operations.
        
#         Args:
#             Q: JPEG quality factor (1-100)
#             subsample: Chroma subsampling (0='4:4:4', 1='4:2:2', 2='4:2:0')
#         """
#         super(DifferentiableJpeg, self).__init__()
#         self.Q = Q
#         self.subsample = subsample
        
#         # Register quantization tables as buffers
#         self.register_buffer('y_table', self._get_quantization_table(Q, 'luma'))
#         self.register_buffer('c_table', self._get_quantization_table(Q, 'chroma'))
        
#         # Register DCT matrix as buffer
#         self.register_buffer('dct_matrix', self._get_dct_matrix())
        
#     def _get_dct_matrix(self):
#         """Create 8x8 DCT transformation matrix."""
#         dct = torch.zeros(8, 8)
#         for i in range(8):
#             for j in range(8):
#                 if i == 0:
#                     dct[i, j] = 1.0 / np.sqrt(8)
#                 else:
#                     dct[i, j] = np.sqrt(2.0 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
#         return dct
    
#     def _get_quantization_table(self, quality, table_type='luma'):
#         """Generate JPEG quantization table based on quality factor."""
#         # Standard JPEG quantization tables
#         if table_type == 'luma':
#             base_table = torch.tensor([
#                 [16, 11, 10, 16, 24, 40, 51, 61],
#                 [12, 12, 14, 19, 26, 58, 60, 55],
#                 [14, 13, 16, 24, 40, 57, 69, 56],
#                 [14, 17, 22, 29, 51, 87, 80, 62],
#                 [18, 22, 37, 56, 68, 109, 103, 77],
#                 [24, 35, 55, 64, 81, 104, 113, 92],
#                 [49, 64, 78, 87, 103, 121, 120, 101],
#                 [72, 92, 95, 98, 112, 100, 103, 99]
#             ], dtype=torch.float32)
#         else:  # chroma
#             base_table = torch.tensor([
#                 [17, 18, 24, 47, 99, 99, 99, 99],
#                 [18, 21, 26, 66, 99, 99, 99, 99],
#                 [24, 26, 56, 99, 99, 99, 99, 99],
#                 [47, 66, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99],
#                 [99, 99, 99, 99, 99, 99, 99, 99]
#             ], dtype=torch.float32)
        
#         # Scale based on quality
#         if quality < 50:
#             scale = 5000 / quality
#         else:
#             scale = 200 - 2 * quality
        
#         table = torch.floor((base_table * scale + 50) / 100)
#         table = torch.clamp(table, 1, 255)
        
#         return table
    
#     def _rgb_to_ycbcr(self, rgb):
#         """Convert RGB to YCbCr color space."""
#         # rgb shape: (B, 3, H, W), values in [0, 1]
#         r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
#         y = 0.299 * r + 0.587 * g + 0.114 * b
#         cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
#         cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        
#         return torch.cat([y, cb, cr], dim=1)
    
#     def _ycbcr_to_rgb(self, ycbcr):
#         """Convert YCbCr to RGB color space."""
#         # ycbcr shape: (B, 3, H, W)
#         y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        
#         cb = cb - 0.5
#         cr = cr - 0.5
        
#         r = y + 1.402 * cr
#         g = y - 0.344136 * cb - 0.714136 * cr
#         b = y + 1.772 * cb
        
#         return torch.cat([r, g, b], dim=1)
    
#     def _chroma_subsample(self, cb, cr):
#         """Apply chroma subsampling."""
#         if self.subsample == 0:  # 4:4:4 - no subsampling
#             return cb, cr
#         elif self.subsample == 1:  # 4:2:2 - horizontal subsampling
#             cb = F.avg_pool2d(cb, kernel_size=(1, 2), stride=(1, 2))
#             cr = F.avg_pool2d(cr, kernel_size=(1, 2), stride=(1, 2))
#         elif self.subsample == 2:  # 4:2:0 - both directions
#             cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
#             cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
        
#         return cb, cr
    
#     def _chroma_upsample(self, cb, cr, target_size):
#         """Upsample chroma channels back to original size."""
#         if self.subsample == 0:
#             return cb, cr
        
#         # Use bilinear interpolation for upsampling
#         cb = F.interpolate(cb, size=target_size, mode='bilinear', align_corners=False)
#         cr = F.interpolate(cr, size=target_size, mode='bilinear', align_corners=False)
        
#         return cb, cr
    
#     def _block_dct(self, channel, q_table):
#         """Apply DCT, quantization, dequantization, and IDCT on 8x8 blocks."""
#         B, C, H, W = channel.shape
        
#         # Pad to make dimensions divisible by 8
#         pad_h = (8 - H % 8) % 8
#         pad_w = (8 - W % 8) % 8
#         channel = F.pad(channel, (0, pad_w, 0, pad_h), mode='reflect')
        
#         _, _, H_pad, W_pad = channel.shape
        
#         # Reshape into 8x8 blocks: (B, C, H/8, 8, W/8, 8)
#         blocks = channel.unfold(2, 8, 8).unfold(3, 8, 8)
#         blocks = blocks.contiguous().view(B, C, -1, 8, 8)
        
#         # Center values around 0 (subtract 128 for 8-bit range [0, 255])
#         blocks = blocks - 0.5
        
#         # Apply DCT: D * block * D^T
#         dct_matrix = self.dct_matrix.to(blocks.device)
#         dct_blocks = torch.matmul(dct_matrix, blocks)
#         dct_blocks = torch.matmul(dct_blocks, dct_matrix.t())
        
#         # Quantize
#         q_table = q_table.to(blocks.device).view(1, 1, 1, 8, 8)
#         quantized = torch.round(dct_blocks / q_table)
        
#         # Dequantize
#         dequantized = quantized * q_table
        
#         # Apply IDCT: D^T * block * D
#         idct_blocks = torch.matmul(dct_matrix.t(), dequantized)
#         idct_blocks = torch.matmul(idct_blocks, dct_matrix)
        
#         # Add back 128
#         idct_blocks = idct_blocks + 0.5
        
#         # Reshape back to image
#         num_blocks_h = H_pad // 8
#         num_blocks_w = W_pad // 8
#         idct_blocks = idct_blocks.view(B, C, num_blocks_h, num_blocks_w, 8, 8)
#         idct_blocks = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
#         result = idct_blocks.view(B, C, H_pad, W_pad)
        
#         # Remove padding
#         if pad_h > 0 or pad_w > 0:
#             result = result[:, :, :H, :W]
        
#         return result
    
#     def forward(self, image_and_cover):
#         """
#         Apply differentiable JPEG compression.
        
#         Args:
#             image_and_cover: Tuple of (image, cover_image)
#                 image shape: (B, 3, H, W), values in [-1, 1]
        
#         Returns:
#             Compressed image with same shape and range as input
#         """
#         image, cover_image = image_and_cover
        
#         # Convert from [-1, 1] to [0, 1]
#         image_01 = (image.clamp(-1, 1) + 1) / 2
        
#         # RGB to YCbCr
#         ycbcr = self._rgb_to_ycbcr(image_01)
#         y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        
#         # Store original size for upsampling
#         _, _, H, W = y.shape
        
#         # Chroma subsampling
#         cb_sub, cr_sub = self._chroma_subsample(cb, cr)
        
#         # Apply DCT + Quantization + IDCT
#         y_compressed = self._block_dct(y, self.y_table)
#         cb_compressed = self._block_dct(cb_sub, self.c_table)
#         cr_compressed = self._block_dct(cr_sub, self.c_table)
        
#         # Chroma upsampling
#         cb_up, cr_up = self._chroma_upsample(cb_compressed, cr_compressed, (H, W))
        
#         # Combine channels
#         ycbcr_compressed = torch.cat([y_compressed, cb_up, cr_up], dim=1)
        
#         # YCbCr to RGB
#         rgb_compressed = self._ycbcr_to_rgb(ycbcr_compressed)
        
#         # Clamp to valid range and convert back to [-1, 1]
#         rgb_compressed = torch.clamp(rgb_compressed, 0, 1)
#         result = rgb_compressed * 2 - 1
        
#         return result

# ---------------------------------------------------------------------------------------


class JpegTest_Random_gradually_decrease(nn.Module):
    def __init__(self,min_quality=50, max_quality=100, total_epochs=100, subsample=2, path="temp/"):

        super(JpegTest_Random_gradually_decrease, self).__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.total_epochs = total_epochs
        self.current_quality = max_quality
        
        self.subsample = subsample
        self.path = path

        if not os.path.exists(path):
            os.mkdir(path)

    def set_epoch(self, epoch=None):
        if epoch is None:
            self.current_quality = self.max_quality
        else:
            if epoch >= self.total_epochs:
                # Once reached total_epochs, stick to min_quality
                self.current_quality = self.min_quality
            else:
                # Gradual decrease from max_quality to min_quality
                self.current_quality = self.max_quality - (self.max_quality - self.min_quality) * (epoch / self.total_epochs)
                self.current_quality = max(self.current_quality, self.min_quality)

    def get_path(self):
        return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        shape = image.shape
        noised_image = torch.zeros_like(image)

        for i in range(shape[0]):
            # Pick Q for this image

            q_value = round(self.current_quality if self.current_quality is not None else 50)
            # print(q_value)

            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_path()
            while os.path.exists(file):
                file = self.get_path()

            im.save(file, format="JPEG", quality=q_value, subsampling=self.subsample)
            jpeg = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)

        return noised_image


class JpegTest_Random(nn.Module):
    def __init__(self, Q=(50, 90), subsample=2, path="temp/", None_randomize_q=False, q_fix=50):
        """
        Q: int, default JPEG quality when randomize_q=False
        subsample: JPEG subsampling value (0, 1, or 2)
        path: temp folder for saving JPEG
        q_range: tuple(min_Q, max_Q) for random JPEG quality
        """
        super(JpegTest_Random, self).__init__()
        self.Q = Q
        self.current_quality = None
        self.subsample = subsample
        self.path = path
        self.None_randomize_q = None_randomize_q
        self.q_fix = q_fix

        if not os.path.exists(path):
            os.mkdir(path)

    def set_epoch(self, epoch=None):
         self.current_quality = random.randint(*self.Q)

    def get_path(self):
        return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        shape = image.shape
        noised_image = torch.zeros_like(image)

        for i in range(shape[0]):
            # Pick Q for this image
            if self.None_randomize_q:
                q_value = self.q_fix
            else:
                q_value = self.current_quality if self.current_quality is not None else 50

            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)

            file = self.get_path()
            while os.path.exists(file):
                file = self.get_path()

            im.save(file, format="JPEG", quality=q_value, subsampling=self.subsample)
            jpeg = np.array(Image.open(file), dtype=np.uint8)
            os.remove(file)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)

        return noised_image

# class JpegTest(nn.Module):
#     def __init__(self, Q=50, subsample=2, path="temp/", randomize_q=False, q_range=(30, 70)):
#         """
#         Q: int, default JPEG quality when randomize_q=False
#         subsample: JPEG subsampling value (0, 1, or 2)
#         path: temp folder for saving JPEG
#         randomize_q: if True, use a random Q in q_range for each image
#         q_range: tuple(min_Q, max_Q) for random JPEG quality
#         """
#         super(JpegTest, self).__init__()
#         self.Q = Q
#         self.subsample = subsample
#         self.path = path
#         self.randomize_q = randomize_q
#         self.q_range = q_range

#         if not os.path.exists(path):
#             os.mkdir(path)

#     def get_path(self):
#         return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

#     def forward(self, image_and_cover):
#         image, cover_image = image_and_cover
#         shape = image.shape
#         noised_image = torch.zeros_like(image)

#         for i in range(shape[0]):
#             # Pick Q for this image
#             if self.randomize_q:
#                 q_value = random.randint(self.q_range[0], self.q_range[1])
#             else:
#                 q_value = self.Q

#             single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
#             im = Image.fromarray(single_image)

#             file = self.get_path()
#             while os.path.exists(file):
#                 file = self.get_path()

#             im.save(file, format="JPEG", quality=q_value, subsampling=self.subsample)
#             jpeg = np.array(Image.open(file), dtype=np.uint8)
#             os.remove(file)

#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])

#             noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)

#         return noised_image

class JpegTest(nn.Module):
	def __init__(self, Q, subsample=2, path="temp/"):
		super(JpegTest, self).__init__()
		self.Q = Q
		self.subsample = subsample
		self.path = path
		if not os.path.exists(path): os.mkdir(path)

	def get_path(self):
		return self.path + ''.join(random.sample(string.ascii_letters + string.digits, 16)) + ".jpg"

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		shape = image.shape
		noised_image = torch.zeros_like(image)

		for i in range(shape[0]):
			single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).to('cpu', torch.uint8).numpy()
			im = Image.fromarray(single_image)

			file = self.get_path()
			while os.path.exists(file):
				file = self.get_path()
			im.save(file, format="JPEG", quality=self.Q, subsampling=self.subsample)
			jpeg = np.array(Image.open(file), dtype=np.uint8)
			os.remove(file)

			transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			])

			noised_image[i] = transform(jpeg).unsqueeze(0).to(image.device)

		return noised_image

class JpegTestOpenCV(nn.Module):
    """OpenCV - Different JPEG encoder (libjpeg-turbo)"""
    def __init__(self, Q):
        super(JpegTestOpenCV, self).__init__()
        self.Q = Q
    
    def forward(self, image):
        """Apply OpenCV JPEG compression"""
        image, cover = image
        shape = image.shape
        noised_image = torch.zeros_like(image)
        
        for i in range(shape[0]):
            # Convert to numpy uint8 (BGR for OpenCV)
            single_image = ((image[i].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
            single_image = cv2.cvtColor(single_image, cv2.COLOR_RGB2BGR)
            
            # JPEG encode/decode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.Q]
            _, encimg = cv2.imencode('.jpg', single_image, encode_param)
            decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
            
            # Convert back to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            noised_image[i] = transform(decimg)
        
        return noised_image

class JpegBasic(nn.Module):
	def __init__(self):
		super(JpegBasic, self).__init__()

	def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

		luminance_quant_tbl = (torch.tensor([
			[16, 11, 10, 16, 24, 40, 51, 61],
			[12, 12, 14, 19, 26, 58, 60, 55],
			[14, 13, 16, 24, 40, 57, 69, 56],
			[14, 17, 22, 29, 51, 87, 80, 62],
			[18, 22, 37, 56, 68, 109, 103, 77],
			[24, 35, 55, 64, 81, 104, 113, 92],
			[49, 64, 78, 87, 103, 121, 120, 101],
			[72, 92, 95, 98, 112, 100, 103, 99]
		], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
			image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

		chrominance_quant_tbl = (torch.tensor([
			[17, 18, 24, 47, 99, 99, 99, 99],
			[18, 21, 26, 66, 99, 99, 99, 99],
			[24, 26, 56, 99, 99, 99, 99, 99],
			[47, 66, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99]
		], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
			image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

		q_image_yuv_dct = image_yuv_dct.clone()
		q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
		q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
		q_image_yuv_dct_round = round_func(q_image_yuv_dct)
		return q_image_yuv_dct_round

	def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

		luminance_quant_tbl = (torch.tensor([
			[16, 11, 10, 16, 24, 40, 51, 61],
			[12, 12, 14, 19, 26, 58, 60, 55],
			[14, 13, 16, 24, 40, 57, 69, 56],
			[14, 17, 22, 29, 51, 87, 80, 62],
			[18, 22, 37, 56, 68, 109, 103, 77],
			[24, 35, 55, 64, 81, 104, 113, 92],
			[49, 64, 78, 87, 103, 121, 120, 101],
			[72, 92, 95, 98, 112, 100, 103, 99]
		], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
			q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

		chrominance_quant_tbl = (torch.tensor([
			[17, 18, 24, 47, 99, 99, 99, 99],
			[18, 21, 26, 66, 99, 99, 99, 99],
			[24, 26, 56, 99, 99, 99, 99, 99],
			[47, 66, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99],
			[99, 99, 99, 99, 99, 99, 99, 99]
		], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
			q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

		image_yuv_dct = q_image_yuv_dct.clone()
		image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
		image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
		return image_yuv_dct

	def dct(self, image):
		# coff for dct and idct
		coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
		coff[0, :] = 1 * np.sqrt(1 / 8)
		for i in range(1, 8):
			for j in range(8):
				coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

		split_num = image.shape[2] // 8
		image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
		image_dct = torch.matmul(coff, image_dct)
		image_dct = torch.matmul(image_dct, coff.permute(1, 0))
		image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

		return image_dct

	def idct(self, image_dct):
		# coff for dct and idct
		coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
		coff[0, :] = 1 * np.sqrt(1 / 8)
		for i in range(1, 8):
			for j in range(8):
				coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

		split_num = image_dct.shape[2] // 8
		image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
		image = torch.matmul(coff.permute(1, 0), image)
		image = torch.matmul(image, coff)
		image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

		return image

	def rgb2yuv(self, image_rgb):
		image_yuv = torch.empty_like(image_rgb)
		image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
								  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
		image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
								  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
		image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
								  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
		return image_yuv

	def yuv2rgb(self, image_yuv):
		image_rgb = torch.empty_like(image_yuv)
		image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
		image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
								  - 0.714103821 * image_yuv[:, 2:3, :, :]
		image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
		return image_rgb

	def yuv_dct(self, image, subsample):
		# clamp and convert from [-1,1] to [0,255]
		image = (image.clamp(-1, 1) + 1) * 255 / 2

		# pad the image so that we can do dct on 8x8 blocks
		pad_height = (8 - image.shape[2] % 8) % 8
		pad_width = (8 - image.shape[3] % 8) % 8
		image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

		# convert to yuv
		image_yuv = self.rgb2yuv(image)

		assert image_yuv.shape[2] % 8 == 0
		assert image_yuv.shape[3] % 8 == 0

		# subsample
		image_subsample = self.subsampling(image_yuv, subsample)

		# apply dct
		image_dct = self.dct(image_subsample)

		return image_dct, pad_width, pad_height

	def idct_rgb(self, image_quantization, pad_width, pad_height):
		# apply inverse dct (idct)
		image_idct = self.idct(image_quantization)

		# transform from yuv to to rgb
		image_ret_padded = self.yuv2rgb(image_idct)

		# un-pad
		image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
					:image_ret_padded.shape[3] - pad_width].clone()

		return image_rgb * 2 / 255 - 1

	def subsampling(self, image, subsample):
		if subsample == 2:
			split_num = image.shape[2] // 8
			image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
			for i in range(8):
				if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
			for j in range(8):
				if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
			image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
		return image


class Jpeg(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(Jpeg, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
		return noised_image.clamp(-1, 1)


class JpegSS(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegSS, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_ss(self, x):
		cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
		return cond * (x ** 3) + (1 - cond) * x

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# quantization
		image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

		# reverse quantization
		image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
		return noised_image.clamp(-1, 1)
     
class JpegMask(JpegBasic):
	def __init__(self, Q, subsample=0):
		super(JpegMask, self).__init__()

		# quantization table
		self.Q = Q
		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

		# subsample
		self.subsample = subsample

	def round_mask(self, x):
		mask = torch.zeros(1, 3, 8, 8).to(x.device)
		mask[:, 0:1, :5, :5] = 1
		mask[:, 1:3, :3, :3] = 1
		mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
		return x * mask

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover

		# [-1,1] to [0,255], rgb2yuv, dct
		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

		# mask
		image_mask = self.round_mask(image_dct)

		# idct, yuv2rgb, [0,255] to [-1,1]
		noised_image = self.idct_rgb(image_mask, pad_width, pad_height)
		return noised_image.clamp(-1, 1)

# class JpegMask(JpegBasic):
#     def __init__(self, q_range=(30, 90), subsample=0):
#         super(JpegMask, self).__init__()
#         self.q_range = q_range
#         self.subsample = subsample

#         # Initialize with some default Q
#         self.Q = random.randint(*self.q_range)
#         self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

#     def set_epoch(self, epoch=None):
#         """Randomize JPEG quality at the start of each epoch"""
#         self.Q = random.randint(*self.q_range)
#         self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

#     def round_mask(self, x):
#         mask = torch.zeros(1, 3, 8, 8).to(x.device)
#         mask[:, 0:1, :5, :5] = 1
#         mask[:, 1:3, :3, :3] = 1
#         mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
#         return x * mask

#     def forward(self, image_and_cover):
#         image, cover_image = image_and_cover

#         # [-1,1] to [0,255], rgb2yuv, dct
#         image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

#         # Apply mask
#         image_mask = self.round_mask(image_dct)

#         # Reverse quantization with current Q
#         image_quantization = self.std_reverse_quantization(image_mask, self.scale_factor)

#         # idct, yuv2rgb, [0,255] to [-1,1]
#         noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
#         return noised_image.clamp(-1, 1)

class JpegMask_gradually_decrease(JpegBasic):
    def __init__(self, min_quality=50, max_quality=100, total_epochs=100, subsample=0):
        super(JpegMask_gradually_decrease, self).__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.total_epochs = total_epochs
        self.subsample = subsample

        # Start with max quality at initialization
        self.current_quality = max_quality
        self.Q = self.current_quality
        self.update_scale_factor()

    def set_epoch(self, epoch=None):
        """Gradually decrease JPEG quality over epochs"""
        if epoch is None:
            # fallback: randomize in range
            self.current_quality = random.randint(self.min_quality, self.max_quality)
        else:
            if epoch >= self.total_epochs:
                self.current_quality = self.min_quality
            else:
                fraction = epoch / self.total_epochs
                self.current_quality = int(self.max_quality - fraction * (self.max_quality - self.min_quality))

        self.Q = self.current_quality
        self.update_scale_factor()

    def update_scale_factor(self):
        """Update scale factor based on current Q"""
        if self.Q >= 50:
            self.scale_factor = 2 - self.Q * 0.02
        else:
            self.scale_factor = 50 / self.Q

    def round_mask(self, x):
        mask = torch.zeros(1, 3, 8, 8, device=x.device)
        mask[:, 0:1, :5, :5] = 1   # Y channel (luminance) more preserved
        mask[:, 1:3, :3, :3] = 1   # UV channels (chrominance) less preserved
        mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
        return x * mask

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        # [-1,1] → [0,255], RGB → YUV → DCT
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        # Apply mask
        image_mask = self.round_mask(image_dct)

        # Reverse quantization with current scale factor
        image_quantization = self.std_reverse_quantization(image_mask, self.scale_factor)

        # IDCT, YUV → RGB, [0,255] → [-1,1]
        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)

        return noised_image.clamp(-1, 1)

    
# class JpegMask(JpegBasic):
# 	def __init__(self, q_range=(30, 90), subsample=0):
# 		super(JpegMask, self).__init__()

# 		# quantization table
# 		self.Q = Q
# 		self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

# 		# subsample
# 		self.subsample = subsample

# 	def round_mask(self, x):
# 		mask = torch.zeros(1, 3, 8, 8).to(x.device)
# 		mask[:, 0:1, :5, :5] = 1
# 		mask[:, 1:3, :3, :3] = 1
# 		mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
# 		return x * mask

# 	def forward(self, image_and_cover):
# 		image, cover_image = image_and_cover

# 		# [-1,1] to [0,255], rgb2yuv, dct
# 		image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

# 		# mask
# 		image_mask = self.round_mask(image_dct)

# 		# idct, yuv2rgb, [0,255] to [-1,1]
# 		noised_image = self.idct_rgb(image_mask, pad_width, pad_height)
# 		return noised_image.clamp(-1, 1)

# -------------------- Jpeg layer from ARWGAN (fixed) --------------------
 
 
def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.
 
    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.
 
    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.
    """
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
    r"""Convert an YCbCr image to RGB.
 
    The image data is assumed to be in the range of (0, 1).
 
    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.
 
    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.
    """
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
 
 
def rgb_to_ycbcr_jpeg(image):
    return rgb_to_ycbcr(image)
 
 
# 2. Chroma subsampling (4:2:0)
def downsampling_420(image):
    # input: batch x 3 x height x width
    # output: tuple of length 3
    #   y:  batch x 1 x height x width
    #   cb: batch x 1 x height/2 x width/2
    #   cr: batch x 1 x height/2 x width/2
    y, cb, cr = torch.split(image, 1, dim=1)
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
    return y, cb, cr
 
 
# 3. Block splitting
def image_to_patches(image):
    # input: batch x c x h x w   (c == 1 per call site)
    # output: batch x h*w/64 x 8 x 8
    k = 8
    b, c, h, w = image.size()
    image_reshaped = torch.reshape(image, [b, h // k, k, -1, k])
    image_transposed = image_reshaped.permute((0, 1, 3, 2, 4))
    return torch.reshape(image_transposed, [b, -1, k, k])
 
 
# 4. DCT
def dct_8x8(image):
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    tensor = torch.from_numpy(tensor)
 
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    scale = np.outer(alpha, alpha) * 0.25
    scale = torch.from_numpy(scale)
 
    image = image - 128.
    result = scale * torch.tensordot(image, tensor, dims=2).reshape(image.size())
    return result
 
 
# 5. Quantization
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
y_table = torch.from_numpy(y_table)
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = torch.from_numpy(c_table)
 
 
def y_quantize(image, rounding, factor=1):
    image = image / (y_table * factor)
    image = rounding(image)
    return image
 
 
def c_quantize(image, rounding, factor=1):
    image = image / (c_table * factor)
    image = rounding(image)
    return image
 
 
# -5. Dequantization
def y_dequantize(image, factor=1):
    return image * (y_table * factor)
 
 
def c_dequantize(image, factor=1):
    return image * (c_table * factor)
 
 
# -4. Inverse DCT
def idct_8x8(image):
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    tensor = torch.from_numpy(tensor)
 
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    alpha = np.outer(alpha, alpha)
    alpha = torch.from_numpy(alpha)
 
    image = image * alpha
    result = 0.25 * torch.tensordot(image, tensor, dims=2).reshape(image.size()) + 128
    return result
 
 
# -3. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x 8 x 8
    # output: batch x height x width
    k = 8
    batch_size = patches.size()[0]
    image_reshaped = torch.reshape(patches,
                                   [batch_size, height // k, width // k, k, k])
    image_transposed = image_reshaped.permute((0, 1, 3, 2, 4))
    return torch.reshape(image_transposed, [batch_size, height, width])
 
 
# -2. Chroma upsampling (4:2:0 -> 4:4:4)
def upsampling_420(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x 3 x height x width
    def repeat(x, k=2):
        # FIX: nearest-neighbor upsample -- each downsampled pixel becomes
        # its own k x k block, correctly aligned with the full-res luma plane.
        #
        # The original implementation used `x.repeat([1, 1, k, k])`, which
        # *tiles* the whole half-resolution plane (pastes a copy of the
        # entire image next to itself) instead of expanding each pixel in
        # place. That silently scrambles which chroma value lands on which
        # luma pixel and is the source of the localized red/color blotches.
        return x.repeat_interleave(k, dim=1).repeat_interleave(k, dim=2)
 
    cb = repeat(cb)
    cr = repeat(cr)
    return torch.stack((y, cb, cr), dim=1)
 
 
# -1. YCbCr -> RGB
def ycbcr_to_rgb_jpeg(image):
    return ycbcr_to_rgb(image)
 
 
def diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3
 
 
def quality_to_factor(quality, min_factor=0.01):
    if quality < 50:
        factor = 50. / quality
    else:
        factor = 2. - quality * 0.02
    return max(factor, min_factor)
 
 
def jpeg_compress_decompress(image, downsample_c=False, rounding=diff_round, factor=1.):
    image = (image + 1) / 2   # [-1,1] -> [0,1]
    image = image * 255       # [0,1] -> [0,255]
    b, c, h, w = image.size()
 
    orig_height, orig_width = h, w
    # FIX: top/left must be defined even when no padding is needed, and must
    # be reused (not recomputed from vpad/wpad alone) when cropping back --
    # otherwise an odd pad amount shifts the whole image on crop.
    top, left = 0, 0
 
    if h % 16 != 0 or w % 16 != 0:
        h = ((h - 1) // 16 + 1) * 16
        w = ((w - 1) // 16 + 1) * 16
        vpad = h - orig_height
        wpad = w - orig_width
        top = vpad // 2
        bottom = vpad - top
        left = wpad // 2
        right = wpad - left
        image = F.pad(image, [left, right, top, bottom])
 
    # --- compression ---
    image = rgb_to_ycbcr_jpeg(image)
    if downsample_c:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = torch.split(image, 1, dim=1)
    components = {'y': y, 'cb': cb, 'cr': cr}
 
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize(comp, rounding, factor)
        components[k] = comp
 
    # --- decompression ---
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr') and downsample_c:
            comp = patches_to_image(comp, h // 2, w // 2)
        else:
            comp = patches_to_image(comp, h, w)
        components[k] = comp
 
    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.stack((y, cb, cr), dim=1)
    image = ycbcr_to_rgb_jpeg(image)
 
    # crop back to original size -- symmetric with the padding applied above
    if orig_height != h or orig_width != w:
        image = image[:, :, top:top + orig_height, left:left + orig_width]
 
    # clamp to [0,255], then normalize to [0,1]
    image = torch.min(torch.tensor(255., device=image.device),
                       torch.max(torch.tensor(0., device=image.device), image))
    image = image / 255.0
 
    # rescale to [-1,1] for model compatibility
    image = image * 2.0 - 1.0
 
    return image
 
 
class JpegARWGAN(nn.Module):
    def __init__(self, factor, chroma_subsample: bool = True):
        """
        Args:
            factor: JPEG quality (0-100 scale, passed through quality_to_factor).
            chroma_subsample: if True, applies 4:2:0 chroma subsampling
                (Cb/Cr downsampled 2x, matching standard JPEG). If False,
                runs 4:4:4 (no chroma subsampling).
        """
        super(JpegARWGAN, self).__init__()
        self.factor = factor
        self.chroma_subsample = chroma_subsample
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
    def forward(self, noise_and_cover):
        encoded_image = noise_and_cover[0]
        encoded_image = encoded_image.cpu()
        jpeg_image = jpeg_compress_decompress(
            encoded_image,
            downsample_c=self.chroma_subsample,
            factor=quality_to_factor(self.factor),
        )
        noise_and_cover[0] = torch.FloatTensor(jpeg_image).to(self.device)
        return noise_and_cover[0]
    
class JpegARWGAN_444(nn.Module):
    def __init__(self,factor):
        super(JpegARWGAN_444, self).__init__()
        self.factor=factor
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encoded_image=noise_and_cover[0]
        encoded_image=encoded_image.cpu()
        jpeg_image=jpeg_compress_decompress(encoded_image, downsample_c=False, factor=quality_to_factor(self.factor))
        noise_and_cover[0]=torch.FloatTensor(jpeg_image).to(self.device)
        return noise_and_cover[0]


# --------------------------------------------

class JpegARWGAN_NEW(nn.Module):
    def __init__(self, Q=50):
        super(JpegARWGAN_NEW, self).__init__()
        # quantization quality
        self.Q = Q
        # scale factor based on JPEG convention
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

    def rounding(self, x):
        return torch.round(x)

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        # clamp [-1,1] → [0,255]
        image = ((image.clamp(-1, 1) + 1) / 2) * 255.0

        b, c, h, w = image.shape
        # convert RGB → YCbCr
        ycbcr = rgb_to_ycbcr(image)

        # chroma subsampling 4:2:0
        y, cb, cr = downsampling_420(ycbcr)

        # split into 8x8 blocks
        y_blocks = image_to_patches(y)
        cb_blocks = image_to_patches(cb)
        cr_blocks = image_to_patches(cr)

        # DCT
        y_dct = dct_8x8(y_blocks)
        cb_dct = dct_8x8(cb_blocks)
        cr_dct = dct_8x8(cr_blocks)

        # quantization
        y_quant = y_quantize(y_dct, self.rounding, self.scale_factor)
        cb_quant = c_quantize(cb_dct, self.rounding, self.scale_factor)
        cr_quant = c_quantize(cr_dct, self.rounding, self.scale_factor)

        # dequantization
        y_dequant = y_dequantize(y_quant, self.scale_factor)
        cb_dequant = c_dequantize(cb_quant, self.scale_factor)
        cr_dequant = c_dequantize(cr_quant, self.scale_factor)

        # IDCT
        y_idct = idct_8x8(y_dequant)
        cb_idct = idct_8x8(cb_dequant)
        cr_idct = idct_8x8(cr_dequant)

        # join patches back
        y_rec = patches_to_image(y_idct, h, w)
        cb_rec = patches_to_image(cb_idct, h // 2, w // 2)
        cr_rec = patches_to_image(cr_idct, h // 2, w // 2)

        # upsample chroma back to full res
        cb_up = F.interpolate(cb_rec, size=(h, w), mode="nearest")
        cr_up = F.interpolate(cr_rec, size=(h, w), mode="nearest")

        # merge back YCbCr → RGB
        rec_ycbcr = torch.cat([y_rec, cb_up, cr_up], dim=1)
        rec_rgb = ycbcr_to_rgb(rec_ycbcr)

        # back to [-1,1]
        return (rec_rgb / 255.0 * 2 - 1).clamp(-1, 1)