import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .crop import Crop, Cropout, Dropout
# from .cropout import CropoutARWGAN
from .gaussian_noise import GN
from .middle_filter import MF
from .gaussian_filter import GF
from .salt_pepper_noise import SP
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest_Random,JpegTest, JpegMask_gradually_decrease, JpegTest_Random_gradually_decrease, DifferentiableJpeg, JpegTestOpenCV, JpegARWGAN
from .jpeg_with_noise_v2 import JpegARWGAN420_v2
from .jpeg_with_noise import JpegARWGAN420_noise
from .combined import Combined
