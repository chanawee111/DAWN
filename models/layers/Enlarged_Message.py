import torch
import torch.nn  as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import numpy as np
from . import *

# from .SE_layer import SENet

class Enlarged_Message(nn.Module):
    def __init__(self, input_dim=64,image_size=128):
        super(Enlarged_Message, self).__init__()
        self.input_dim = input_dim
        
        # Fully connected layer to expand the secret vector
        if input_dim == 64:
            self.fc = nn.Linear(input_dim, 64 * 64 * 3)
        else:
            self.fc = nn.Linear(input_dim, 32 * 32 * 3)
        # self.fc = nn.Linear(input_dim, 32 * 32 * 3)
        
        # Smoother upsampling
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
        

    def forward(self, secret_bits):
        secret_bits = secret_bits * 2 - 1
        
        # Expand to (B, 3, 50, 50)
        x = self.fc(secret_bits)

        if self.input_dim == 64:
            x = x.view(-1, 3, 64, 64)
        else:
            x = x.view(-1, 3, 32, 32)
        # x = x.view(-1, 3, 32, 32)
        
        # Upsample to (B, 3, 128, 128)
        x = self.upsample(x)
        
        x = torch.clamp(x, -1, 1)
        
        return x

class Enlarged_Message_DTCWT_learnable_upsample_first(nn.Module):
    """
    Expanded secret -> small feature map -> upsample -> DTCWT refine -> output.
    Supports learnable weights for both lowpass and highpass components.

    Args:
        input_dim: number of secret bits (e.g. 30)
        base_map_size: initial FC output spatial size (default 32)
        image_size: final target image size (e.g. 128)
        J: number of DTCWT scales used for refinement
    """
    def __init__(self, input_dim=30, base_map_size=32, image_size=128, J=2,
                 in_channels=3, biort='near_sym_b', qshift='qshift_b'):
        super(Enlarged_Message_DTCWT_learnable_upsample_first, self).__init__()
        assert base_map_size >= 8, "Choose reasonable base_map_size >= 8"

        self.input_dim = input_dim
        self.base_map_size = base_map_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.J = J

        # 1️⃣ Fully connected expansion to small feature map
        self.fc = nn.Linear(input_dim, in_channels * base_map_size * base_map_size)

        # 2️⃣ Upsampling before DTCWT (operate at full resolution)
        if image_size != base_map_size:
            self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

        # 3️⃣ DTCWT forward/inverse setup
        self.xfm = DTCWTForward(J=J, biort=biort, qshift=qshift)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift)

        # 4️⃣ Learnable weights for lowpass and highpass
        self.lowpass_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.lowpass_bias   = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        self.highpass_weights = nn.ParameterList()
        self.highpass_biases = nn.ParameterList()
        for level in range(J):
            gain = nn.Parameter(torch.ones(1, in_channels, 6, 1, 1, 1))
            bias = nn.Parameter(torch.zeros(1, in_channels, 6, 1, 1, 1))
            self.highpass_weights.append(gain)
            self.highpass_biases.append(bias)

        # 5️⃣ Optional refinement conv after inverse
        # self.refine_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        #     nn.Tanh(),
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        # )

    def forward(self, secret_bits):
        """
        secret_bits: (B, input_dim) assumed in {0,1} or [0,1] range.
        returns: (B, in_channels, image_size, image_size) in [-1,1]
        """
        # Normalize [0,1] -> [-1,1]
        x = secret_bits.float() * 2.0 - 1.0

        # FC expansion and reshape
        x = self.fc(x)
        x = x.view(-1, self.in_channels, self.base_map_size, self.base_map_size)

        # Upsample before wavelet transform
        x = self.upsample(x)  # (B, C, image_size, image_size)

        # Forward DTCWT
        Yl, Yh = self.xfm(x)

        # Apply learnable weights to lowpass
        Yl_mod = Yl * self.lowpass_weight + self.lowpass_bias

        # Apply learnable weights to each highpass level
        Yh_mod = []
        for j in range(self.J):
            gain = self.highpass_weights[j]
            bias = self.highpass_biases[j]
            yh_mod = Yh[j] * gain + bias
            Yh_mod.append(yh_mod)

        # Inverse transform
        x_rec = self.ifm((Yl_mod, Yh_mod))

        # Optional refinement
        # x_refined = self.refine_conv(x_rec)

        # Clamp to [-1,1]
        x_out = torch.clamp(x_rec, -1.0, 1.0)
        return x_out
    
# class Enlarged_Message_Transpose(nn.Module):
#     def __init__(self, input_dim=30, base_map_size=32, image_size=128,
#                  in_channels=3, blocks=4, channels=64):
#         super(Enlarged_Message_Transpose, self).__init__()
#         self.H = image_size
#         self.W = image_size
#         self.input_dim = input_dim

#         message_convT_blocks = int(np.log2(self.H // int(np.sqrt(self.input_dim))))
#         message_se_blocks = max(blocks - message_convT_blocks, 1)

#         self.message_pre_layer = nn.Sequential(
#             ConvBNRelu(1, channels),
#             ExpandNet(channels, channels, blocks=message_convT_blocks),
#             SENet(channels, channels, blocks=message_se_blocks),
#         )

#         self.message_first_layer = nn.Sequential(
#             nn.Conv2d(channels, 3, kernel_size=3, padding=1),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(channels // 2, 3, kernel_size=3, padding=1),
#             # nn.Tanh()
#         )

#     def forward(self, secret_bits):
#         secret_bits = secret_bits #* 2.0 - 1.0  
#         padded = F.pad(secret_bits, (0, 6))
#         size = int(np.sqrt(padded.shape[1]))
#         message_image = padded.view(-1, 1, size, size)

#         message_pre = self.message_pre_layer(message_image)
#         enlarged_message = self.message_first_layer(message_pre)

#         # 🚀 Ensure final resolution is exactly [B, 3, 128, 128]
#         if enlarged_message.shape[-1] != self.H:
#             enlarged_message = F.interpolate(
#                 enlarged_message, size=(self.H, self.W),
#                 mode='bilinear', align_corners=False
#             )

#         # print(f"Shape of enlarged_message (after upscale): {enlarged_message.shape}")
#         return enlarged_message


# class Enlarged_Message_Transpose(nn.Module):
#     def __init__(self, input_dim=64, base_map_size=32, image_size=128,
#                  in_channels=3, blocks=4, channels=64):
#         super(Enlarged_Message_Transpose, self).__init__()
#         self.H = image_size
#         self.W = image_size
#         self.input_dim = input_dim
#         self.in_channels = in_channels
#         self.base_map_size = base_map_size

#         # Fully connected layer → project 1D secret into spatial feature map
#         self.fc = nn.Linear(input_dim, in_channels * base_map_size * base_map_size)

#         message_convT_blocks = int(np.log2(self.H // base_map_size))
#         message_se_blocks = max(blocks - message_convT_blocks, 1)

#         self.message_pre_layer = nn.Sequential(
#             ConvBNRelu(in_channels, channels),
#             ExpandNet(channels, channels, blocks=message_convT_blocks),
#             SENet(channels, channels, blocks=message_se_blocks),
#         )

#         self.message_first_layer = nn.Sequential(
#             nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 2, 3, kernel_size=3, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, secret_bits):
#         # Normalize [0,1] → [-1,1]
#         secret_bits = secret_bits * 2.0 - 1.0

#         # 🔹 Project flat secret bits into spatial feature map via Linear
#         x = self.fc(secret_bits)
#         x = x.view(-1, self.in_channels, self.base_map_size, self.base_map_size)
#         # print(f"Shape after fc + view: {x.shape}")  # e.g. [4, 3, 32, 32]

#         # 🔹 CNN and SE layers to enlarge and refine
#         message_pre = self.message_pre_layer(x)
#         # print(f"Shape of message_pre after pre_layer: {message_pre.shape}")

#         enlarged_message = self.message_first_layer(message_pre)
#         # print(f"Shape before final upsample: {enlarged_message.shape}")

#         # 🔹 Final upscale to exactly [B, 3, 128, 128]
#         if enlarged_message.shape[-1] != self.H:
#             enlarged_message = F.interpolate(
#                 enlarged_message, size=(self.H, self.W),
#                 mode='bilinear', align_corners=False
#             )

#         # print(f"Shape of enlarged_message (final): {enlarged_message.shape}")
#         return enlarged_message


# class Enlarged_Message_Transpose(nn.Module):
#     """
#     Upsamples binary secret bits [B, 30] to [B, 3, 128, 128],
#     using ConvTranspose2d + SENet blocks.
#     """

#     def __init__(self, input_dim=30, channels=64, image_size=128, se_blocks=2):
#         super(Enlarged_Message_Transpose, self).__init__()

#         # 1️⃣ Fully connected projection to [B, C, 8, 8]
#         self.fc = nn.Linear(input_dim, channels * 8 * 8)

#         # 2️⃣ Progressive upsampling (ConvTranspose2d + SENet)
#         self.up1 = nn.Sequential(
#             ConvTBNRelu(channels, channels // 2),       # 8x8 → 16x16
#             SENet(channels // 2, channels // 2, blocks=se_blocks)
#         )

#         self.up2 = nn.Sequential(
#             ConvTBNRelu(channels // 2, channels // 4),  # 16x16 → 32x32
#             SENet(channels // 4, channels // 4, blocks=se_blocks)
#         )

#         self.up3 = nn.Sequential(
#             ConvTBNRelu(channels // 4, channels // 8),  # 32x32 → 64x64
#             SENet(channels // 8, channels // 8, blocks=se_blocks)
#         )

#         self.up4 = nn.Sequential(
#             ConvTBNRelu(channels // 8, channels // 16), # 64x64 → 128x128
#             SENet(channels // 16, channels // 16, blocks=se_blocks)
#         )

#         # 3️⃣ Output RGB image
#         self.to_rgb = nn.Sequential(
#             nn.Conv2d(channels // 16, 3, kernel_size=3, stride=1, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, secret_bits):
#         # Normalize binary [0,1] → [-1,1]
#         secret_bits = secret_bits * 2.0 - 1.0  # [B, 30]

#         # FC projection
#         x = self.fc(secret_bits)               # [B, 64*8*8]
#         x = x.view(-1, 64, 8, 8)              # [B, 64, 8, 8]

#         # Upsampling stages
#         x = self.up1(x)   # → 16×16
#         x = self.up2(x)   # → 32×32
#         x = self.up3(x)   # → 64×64
#         x = self.up4(x)   # → 128×128

#         # Output RGB image
#         enlarged_message = self.to_rgb(x)
#         return enlarged_message
