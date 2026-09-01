from . import *

def get_warmup_decoder_weights(
    epoch,
    total_epochs=65,
    encoder_warmup_epochs=3
):
    # Decoder is active from the beginning
    decoder_w = 1.0

    # Encoder warm-up
    if epoch < encoder_warmup_epochs:
        encoder_w = 0.0
    else:
        encoder_w = 1.0

    # Discriminator stays fixed
    discriminator_w = 0.001

    return decoder_w, encoder_w, discriminator_w

