from . import *

def compute_jpeg_diff_map_differentiable(cover_image, jpeg_sim_layer):
    """
    cover_image: (B, 3, H, W) tensor in [-1, 1]
    jpeg_sim_layer: existing JpegARWGAN420_v2 instance
    """
    cover_image_list = [cover_image, 0]

    with torch.no_grad():  
        compressed = jpeg_sim_layer(cover_image_list)  
        diff = (cover_image - compressed).abs()
    return diff 

class JPEGAwareAttentionGate(nn.Module):
    """
    Two-input attention gate: gates an encoder feature stream using a JPEG-4:2:0 
    compression-sensitivity map as the conditioning signal, following the Attention U-Net 
    formulation (Oktay et al., 2018), with the gating signal replaced by a compression-diff prior.
    """
    def __init__(self, feature_channels, diff_channels=3, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or feature_channels // 2

        self.theta_x = nn.Conv2d(feature_channels, inter_channels, kernel_size=1)
        self.theta_diff = nn.Conv2d(diff_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature, diff_map):
        # print(f"Feature shape: {feature.shape}, Diff map shape: {diff_map.shape}")
        x1 = self.theta_x(feature)
        g1 = self.theta_diff(diff_map)
        # print(f"x1 shape: {x1.shape}, g1 shape: {g1.shape}")
        combined = self.relu(x1 + g1)
        alpha = self.psi(combined)          
        return feature * alpha, alpha        