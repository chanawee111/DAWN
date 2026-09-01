from . import *
from .encoder import Encoder
from .decoder import Decoder

from .Noise import Noise
from .noise_layers import JpegARWGAN420_v2

class Encoder_Decoder(nn.Module):
    def __init__(self, image_height, image_width, message_length, noise_layers,batch_size,device=None):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(image_height, image_width, message_length=message_length,batch=batch_size)
        self.decoder = Decoder(image_height, image_width, message_length=message_length)
        self.noise = Noise(noise_layers)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.hint_decay_epochs = 5  # Number of epochs over which to decay hint usage
        # self.alpha = 0.5

        self.jpeg_sim_layer = JpegARWGAN420_v2(quality=50)  # Initialize the JPEG simulation layer

    def forward(self, cover_image, secret_message, epoch, decoder_loss=None):

        # ----------------------------------------------
        # |         Simulate JPEG Artifacts            |
        # ---------------------------------------------- 
        diff_jpeg_map = compute_jpeg_diff_map_differentiable(cover_image, self.jpeg_sim_layer)  # Compute the JPEG difference map

        # ----------------------------------------------
        # |                  Encoder                   |
        # ---------------------------------------------- 
        encoded_image, residual, Enlarged_Message, alpha_list = self.encoder(cover_image, secret_message, diff_jpeg_map)
        # check data range of encoded_image and cover_image
        # print(f"Encoded image range: {encoded_image.min().item()} to {encoded_image.max().item()}")

        # ----------------------------------------------
        # |                  Noise layer               |
        # ---------------------------------------------- 
        noisy_image = self.noise([encoded_image, cover_image])
        # check data range of noisy_image
        # print(f"Noisy image range: {noisy_image.min().item()} to {noisy_image.max().item()}")
 
        # -----------------------------------------------
        # |                  Decoder                    |
        # -----------------------------------------------
        # alpha = max(0.0, 1.0 - epoch / self.hint_decay_epochs)
        # guided_residual = residual * self.alpha if self.training else None

        decoded_message = self.decoder(noisy_image)
        
        return encoded_image, noisy_image, decoded_message
    

