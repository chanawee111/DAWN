from sympy import false

from . import *

class Encoder(nn.Module):

    def __init__(self, H, W, message_length=64, batch=4, supervision=False):
        super(Encoder, self).__init__()
        self.H = H
        self.W = W 
        self.supervision = supervision
        self.message_length = message_length
        self.batch = batch

        self.filters = (16, 32, 64, 128)
        self.cat_channels = self.filters[0]   # 16
        self.cat_blocks = len(self.filters)   # 4
        self.upsample_channels = self.cat_blocks * self.cat_channels  # 4 × 16 = 64


        self.enlarged_message = Enlarged_Message(input_dim=message_length)
        for p in self.enlarged_message.parameters():
            p.requires_grad = True


        # Encoder 
        self.e1 = ConvBlock(3, self.filters[0])  # 3   → 16
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(self.filters[0]*2, self.filters[1], n=2))  # 32 → 32
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(self.filters[1]*2, self.filters[2], n=2))  # 64 → 64
        self.e4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(self.filters[2]*2, self.filters[3], n=2))  # 128 → 128

        # ── Bottleneck ───────────────────────────────────────────
        # e4 after msg inject = filters[3]+filters[3] = 256
        self.bottleneck = nn.Conv2d(
            in_channels=self.filters[3]*2,   # 256
            out_channels=self.filters[3],       # 128
            kernel_size=1
        )

        # d4: 4 inputs → 4×16 = 64 = upsample_channels
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0]*2, self.cat_channels, n=1),  # e1+msg  32 → 16
            ConvBlock(self.filters[1]*2, self.cat_channels, n=1),  # e2+msg  64 → 16
            ConvBlock(self.filters[2]*2, self.cat_channels, n=1),  # e3+msg 128 → 16
            ConvBlock(256, self.cat_channels, n=1),  # e4_injected 256 → 16
            ConvBlock(128,   self.cat_channels, n=1),  # btn    256 → 16
            ConvBlock(3, 3, n=1)
        ])
        self.d4_conv = ConvBlock(83, self.upsample_channels, n=1)      # 80→64

        # d3: 4 inputs → 4×16 = 64
        self.d3 = nn.ModuleList([
            ConvBlock(self.filters[0]*2,      self.cat_channels, n=1),  # e1+msg  32 → 16
            ConvBlock(self.filters[1]*2,      self.cat_channels, n=1),  # e2+msg  64 → 16
            ConvBlock(self.filters[2]*2,      self.cat_channels, n=1),  # e3+msg 128 → 16
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),  # d4      64 → 16
            ConvBlock(3, 3, n=1)
        ])
        self.d3_conv = ConvBlock(67, self.upsample_channels, n=1)  # 64→64

        # d2: 4 inputs → 4×16 = 64
        self.d2 = nn.ModuleList([
            ConvBlock(self.filters[0]*2,      self.cat_channels, n=1),  # e1+msg  32 → 16
            ConvBlock(self.filters[1]*2,      self.cat_channels, n=1),  # e2+msg  64 → 16
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),  # d3      64 → 16
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),  # d4      64 → 16
            ConvBlock(3, 3, n=1)
        ])
        self.d2_conv = ConvBlock(67, self.upsample_channels, n=1)   # 64→64

        # d1: 4 inputs → 4×16 = 64
        self.d1 = nn.ModuleList([
            ConvBlock(self.filters[0]*2,      self.cat_channels, n=1),  # e1+msg  32 → 16
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),  # d2      64 → 16
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),  # d3      64 → 16
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),  # d4      64 → 16
            ConvBlock(3, 3, n=1)
        ])
        self.d1_conv = ConvBlock(67, self.upsample_channels, n=1)   # 64→64

        # ── Final layers ─────────────────────────────────────────
        self.final_1 = nn.Conv2d(self.upsample_channels, self.upsample_channels, kernel_size=1)  # 64→64
        self.final_2 = nn.Conv2d(self.upsample_channels, 64, kernel_size=3, padding=1)           # 64→64
        self.final_3 = nn.Conv2d(64+3, 3, kernel_size=1)                                         # 67→3
        # ```


        # Layer for message injection using Conv2d to transform and downsample the message
        self.injection_layer_e1 = nn.Sequential(
            nn.Conv2d(3, self.filters[0], kernel_size=3, padding=1),   # channel expand, no downsampling
        )
        self.injection_layer_e2 = nn.Sequential(
            nn.Conv2d(3, self.filters[1], kernel_size=3, stride=2, padding=1),  # downsample by 2×
        )
        self.injection_layer_e3 = nn.Sequential(
            nn.Conv2d(3, self.filters[2], kernel_size=3, stride=4, padding=1),  # downsample by 4×
        )
        self.injection_layer_e4 = nn.Sequential(
            nn.Conv2d(3, self.filters[3], kernel_size=3, stride=8, padding=1),  # downsample by 8×
        )

        self.jpeg_gate_e1 = JPEGAwareAttentionGate(feature_channels=self.filters[0], diff_channels=3)
        self.jpeg_gate_e2 = JPEGAwareAttentionGate(feature_channels=self.filters[1], diff_channels=3)
        self.jpeg_gate_e3 = JPEGAwareAttentionGate(feature_channels=self.filters[2], diff_channels=3)
        self.jpeg_gate_e4 = JPEGAwareAttentionGate(feature_channels=self.filters[3], diff_channels=3)
       
        
    def forward(self, cover_image, message, jpeg_diff_map=None):

        Enlarged_Message = self.enlarged_message(message)

        # downsampling image size for concat with fusion edges, 
        # size 128*128 to 16*16
        cover_image_d4 = F.avg_pool2d(cover_image, kernel_size=8, stride=8)
        # size 128*128 to 32*32
        cover_image_d3 = F.avg_pool2d(cover_image, kernel_size=4, stride=4)
        # size 128*128 to 64*64
        cover_image_d2 = F.avg_pool2d(cover_image, kernel_size=2, stride=2)

        fm_e1 = self.injection_layer_e1(Enlarged_Message)
        fm_e2 = self.injection_layer_e2(Enlarged_Message)
        fm_e3 = self.injection_layer_e3(Enlarged_Message)
        fm_e4 = self.injection_layer_e4(Enlarged_Message)

        if jpeg_diff_map is not None:
            diff_e1 = F.adaptive_avg_pool2d(jpeg_diff_map, output_size=fm_e1.shape[-2:])
            # print(f"Diff e1 shape: {diff_e1.shape}, fm_e1 shape: {fm_e1.shape}")
            diff_e2 = F.adaptive_avg_pool2d(jpeg_diff_map, output_size=fm_e2.shape[-2:])
            # print(f"Diff e2 shape: {diff_e2.shape}, fm_e2 shape: {fm_e2.shape}")
            diff_e3 = F.adaptive_avg_pool2d(jpeg_diff_map, output_size=fm_e3.shape[-2:])
            # print(f"Diff e3 shape: {diff_e3.shape}, fm_e3 shape: {fm_e3.shape}")
            diff_e4 = F.adaptive_avg_pool2d(jpeg_diff_map, output_size=fm_e4.shape[-2:])
            # print(f"Diff e4 shape: {diff_e4.shape}, fm_e4 shape: {fm_e4.shape}")

            fm_e1, alpha_e1 = self.jpeg_gate_e1(fm_e1, diff_e1)
            fm_e2, alpha_e2 = self.jpeg_gate_e2(fm_e2, diff_e2)
            fm_e3, alpha_e3 = self.jpeg_gate_e3(fm_e3, diff_e3)
            fm_e4, alpha_e4 = self.jpeg_gate_e4(fm_e4, diff_e4)

        # Encoder — raw feature maps
        e1 = self.e1(cover_image)
        e1_injected = torch.cat([e1, fm_e1], dim=1)
        e2 = self.e2(e1_injected)
        e2_injected = torch.cat([e2, fm_e2], dim=1)
        e3 = self.e3(e2_injected)
        e3_injected = torch.cat([e3, fm_e3], dim=1)
        e4 = self.e4(e3_injected)
        e4_injected = torch.cat([e4, fm_e4], dim=1)
    
        btn = self.bottleneck(e4_injected)

        # d4 → 16×16
        d4_inputs = [
            F.max_pool2d(e1_injected, 8),
            F.max_pool2d(e2_injected, 4),
            F.max_pool2d(e3_injected, 2),
            e4_injected,
            btn,
            cover_image_d4
        ]
        d4 = torch.cat([conv(d) for conv, d in zip(self.d4, d4_inputs)], dim=1)
        d4 = self.d4_conv(d4)

        # d3 → 32×32
        d3_inputs = [
            F.max_pool2d(e1_injected, 4),
            F.max_pool2d(e2_injected, 2),
            e3_injected,
            F.interpolate(d4,  scale_factor=2, mode='bilinear', align_corners=True),
            cover_image_d3
        ]
        d3 = torch.cat([conv(d) for conv, d in zip(self.d3, d3_inputs)], dim=1)
        d3 = self.d3_conv(d3)

        # d2 → 64×64
        d2_inputs = [
            F.max_pool2d(e1_injected, 2),
            e2_injected,
            F.interpolate(d3,  scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d4,  scale_factor=4, mode='bilinear', align_corners=True),
            cover_image_d2
        ]
        d2 = torch.cat([conv(d) for conv, d in zip(self.d2, d2_inputs)], dim=1)
        d2 = self.d2_conv(d2)

        # d1 → 128×128  ← KEY FIX: e1_injected replaces raw e1
        d1_inputs = [
            e1_injected,                                                               # was: e1
            F.interpolate(d2,  scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d3,  scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(d4,  scale_factor=8, mode='bilinear', align_corners=True),
            cover_image
        ]
        d1 = torch.cat([conv(d) for conv, d in zip(self.d1, d1_inputs)], dim=1)
        d1 = self.d1_conv(d1)

        output = self.final_1(d1)
        output = self.final_2(output)
        output = torch.cat([output, cover_image], dim=1)
        output = self.final_3(output)
        output = torch.clamp(output, -1, 1)
        encoded_image = output.clone()

        return output, encoded_image, Enlarged_Message, [alpha_e1,alpha_e2,alpha_e3,alpha_e4]


