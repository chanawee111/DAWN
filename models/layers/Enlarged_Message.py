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

