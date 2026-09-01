import torch
import math

def reshape_binary_message(binary_message):
    """
    Reshape a binary message into a square matrix.
    
    Args:
        binary_message (str): A string of binary digits (0s and 1s)
        
    Returns:
        torch.Tensor: A square matrix of shape (sqrt(L), sqrt(L))
                      where L is the length of the binary message
    """
    # Convert binary string to list of integers
    binary_list = [int(bit) for bit in binary_message]
    
    # Calculate the length of the message
    L = len(binary_list)
    
    # Calculate the square root of L
    sqrt_L = int(math.sqrt(L))
    
    # Check if L is a perfect square
    if sqrt_L * sqrt_L != L:
        # If not a perfect square, pad with zeros
        padded_length = (sqrt_L + 1) * (sqrt_L + 1)
        binary_list.extend([0] * (padded_length - L))
        sqrt_L += 1
    
    # Convert list to tensor
    binary_tensor = torch.tensor(binary_list, dtype=torch.int8)
    
    # Reshape tensor to square matrix
    square_matrix = binary_tensor.reshape(sqrt_L, sqrt_L)
    
    return square_matrix