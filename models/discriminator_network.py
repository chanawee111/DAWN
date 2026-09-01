from . import *

class PatchDiscriminator(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels*2, channels*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels*4, 1, 3, stride=1, padding=1)  
        )

    def forward(self, x):
        return self.model(x)  # shape [B, 1, H/8, W/8]

	
	