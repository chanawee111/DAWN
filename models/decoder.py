from . import *

class Decoder(nn.Module):
    def __init__(self, H, W, message_length, blocks=4, channels=64, High_channel=128, device=None, dropout=0.2):
        super(Decoder, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ============== Single Sequential Feature Extractor ==============
        # Combines what were branch1 -> branch2 -> branch3 into one path:
        # full-res convs, then downsample to H/2 x W/2, then to H/4 x W/4.
        self.layers = nn.Sequential(
            ConvBNRelu(3, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels, High_channel, stride=2),        # -> H/2, W/2
            ConvBNRelu(High_channel, High_channel),
            ConvBNRelu(High_channel, High_channel, stride=2),    # -> H/4, W/4
            ConvBNRelu(High_channel, High_channel // 2),
        )

        # ============== Feature Reduction ==============
        self.reduce = ConvBNRelu(High_channel // 2, 1)

        # ============== Fully-connected head ==============
        self.feat_h = H // 4
        self.feat_w = W // 4
        self.fc = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.feat_h * self.feat_w, (self.feat_h * self.feat_w) // 2),
            nn.Dropout(dropout),
            nn.Linear((self.feat_h * self.feat_w) // 2, message_length)
        )

    def forward(self, image):
        # print(f"Decoder input image range: {image.min().item()} to {image.max().item()}")
        x = self.layers(image)                  # (B, High_channel//2, H/4, W/4)
        x = self.reduce(x)                       # (B, 1, H/4, W/4)
        x = x.view(x.shape[0], -1)                # (B, (H/4)*(W/4))
        message = self.fc(x)                      # (B, message_length)
        return message

