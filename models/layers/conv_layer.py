from models import layers
import torch.nn as nn


class ConvTranspose_BN_Relu(nn.Module):
	"""
	A sequence of Transpose Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1):
		super(ConvTranspose_BN_Relu, self).__init__()

		self.layers = nn.Sequential(
			nn.ConvTranspose2d(channels_in, channels_out, kernel_size, stride, padding),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


# class ConvBNRelu(nn.Module):
# 	"""
# 	A sequence of Convolution, Batch Normalization, and ReLU activation
# 	"""

# 	def __init__(self, channels_in, channels_out, kernel_size=3, stride=1):
# 		super(ConvBNRelu, self).__init__()

# 		self.layers = nn.Sequential(
# 			nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=1),
# 			nn.BatchNorm2d(channels_out),
# 			nn.ReLU(inplace=True)
# 		)

# 	def forward(self, x):
# 		return self.layers(x)
class ConvBNRelu(nn.Module):
    """
    A sequence of Convolution, optional Batch Normalization, and ReLU activation.
    """

    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, use_bn=True, use_relu=True):
        super(ConvBNRelu, self).__init__()

        layers = [nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=padding)]

        if use_bn:
            layers.append(nn.BatchNorm2d(channels_out))

        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
	
class ConvBlock(nn.Module):
    """Custom Conv Block:
    Apply n convolutions with optional BN and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                 padding='same', is_bn=True, is_relu=True, n=2):
        super(ConvBlock, self).__init__()
        
        layers = []
        for i in range(1, n + 1):
            conv = nn.Conv2d(
                in_channels=in_channels if i == 1 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not is_bn
            )
            layers.append(conv)
            
            if is_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            
            if is_relu:
                layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class ConvNet(nn.Module):
	'''
	Network that composed by layers of ConvBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ConvNet, self).__init__()

		layers = [ConvBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

