import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

class YCbCrWeightedMSELoss(nn.Module):
    """
    MSE loss computed separately on Y (luma) and Cb/Cr (chroma) channels,
    then combined with independent weights (alway 1.0).
    """

    def __init__(self, y_weight=1.0, chroma_weight=1.0):
        super().__init__()
        self.y_weight = y_weight
        self.chroma_weight = chroma_weight

        # ITU-R BT.601 RGB->YCbCr coefficients
        self.register_buffer("_r_y", torch.tensor(0.299))
        self.register_buffer("_g_y", torch.tensor(0.587))
        self.register_buffer("_b_y", torch.tensor(0.114))

        self.register_buffer("_r_cb", torch.tensor(-0.168736))
        self.register_buffer("_g_cb", torch.tensor(-0.331264))
        self.register_buffer("_b_cb", torch.tensor(0.5))

        self.register_buffer("_r_cr", torch.tensor(0.5))
        self.register_buffer("_g_cr", torch.tensor(-0.418688))
        self.register_buffer("_b_cr", torch.tensor(-0.081312))

    def _rgb_to_ycbcr(self, image):

        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        y = self._r_y * r + self._g_y * g + self._b_y * b
        cb = self._r_cb * r + self._g_cb * g + self._b_cb * b
        cr = self._r_cr * r + self._g_cr * g + self._b_cr * b
        return y, cb, cr

    def forward(self, encoded, cover):

        y_e, cb_e, cr_e = self._rgb_to_ycbcr(encoded)
        y_c, cb_c, cr_c = self._rgb_to_ycbcr(cover)

        loss_y = F.mse_loss(y_e, y_c)
        loss_c = F.mse_loss(cb_e, cb_c) + F.mse_loss(cr_e, cr_c)

        return self.y_weight * loss_y + self.chroma_weight * loss_c