import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Edge Penalty Loss Functions ====================

def edge_penalty_loss_simple(stego_image, cover_image, edge_width=2):
    """
    Simple edge penalty: Penalize modifications at image borders
    
    Args:
        stego_image: Encoded image [B, C, H, W] in range [-1, 1]
        cover_image: Original cover image [B, C, H, W] in range [-1, 1]
        edge_width: Number of pixels to consider as edge (default: 2)
    
    Returns:
        edge_loss: Mean absolute difference at edges
    """
    # Compute absolute difference
    diff = torch.abs(stego_image - cover_image)
    
    # Extract edge pixels
    top_edge = diff[:, :, :edge_width, :]          # Top edge_width rows
    bottom_edge = diff[:, :, -edge_width:, :]      # Bottom edge_width rows
    left_edge = diff[:, :, :, :edge_width]         # Left edge_width columns
    right_edge = diff[:, :, :, -edge_width:]       # Right edge_width columns
    
    # Average across all edges
    edge_loss = (top_edge.mean() + bottom_edge.mean() + 
                 left_edge.mean() + right_edge.mean()) / 4.0
    
    return edge_loss


def edge_penalty_loss_weighted(stego_image, cover_image, edge_width=4):
    """
    Weighted edge penalty: Stronger penalty for outer edges
    
    Args:
        stego_image: Encoded image [B, C, H, W] in range [-1, 1]
        cover_image: Original cover image [B, C, H, W] in range [-1, 1]
        edge_width: Number of pixels to consider as edge (default: 4)
    
    Returns:
        edge_loss: Weighted edge penalty
    """
    diff = torch.abs(stego_image - cover_image)
    
    # Create distance-based weights (outer pixels get higher penalty)
    # Weight: [edge_width, edge_width-1, ..., 2, 1]
    weights = torch.arange(edge_width, 0, -1, dtype=torch.float32, device=diff.device)
    weights = weights / weights.sum()  # Normalize
    
    total_loss = 0.0
    
    # Top edge (weighted)
    for i in range(edge_width):
        total_loss += weights[i] * diff[:, :, i, :].mean()
    
    # Bottom edge (weighted)
    for i in range(edge_width):
        total_loss += weights[i] * diff[:, :, -(i+1), :].mean()
    
    # Left edge (weighted)
    for i in range(edge_width):
        total_loss += weights[i] * diff[:, :, :, i].mean()
    
    # Right edge (weighted)
    for i in range(edge_width):
        total_loss += weights[i] * diff[:, :, :, -(i+1)].mean()
    
    return total_loss / 4.0


def edge_penalty_loss_l2(stego_image, cover_image, edge_width=2):
    """
    L2-based edge penalty: Uses squared differences
    
    Args:
        stego_image: Encoded image [B, C, H, W]
        cover_image: Original cover image [B, C, H, W]
        edge_width: Number of pixels to consider as edge
    
    Returns:
        edge_loss: Mean squared difference at edges
    """
    # Compute squared difference
    diff_squared = (stego_image - cover_image) ** 2
    
    # Extract edges
    top_edge = diff_squared[:, :, :edge_width, :]
    bottom_edge = diff_squared[:, :, -edge_width:, :]
    left_edge = diff_squared[:, :, :, :edge_width]
    right_edge = diff_squared[:, :, :, -edge_width:]
    
    # MSE at edges
    edge_loss = (top_edge.mean() + bottom_edge.mean() + 
                 left_edge.mean() + right_edge.mean()) / 4.0
    
    return edge_loss


def edge_gradient_penalty(stego_image, cover_image):
    """
    Gradient-based edge penalty: Penalizes sharp changes near edges
    This prevents visible "lines" at borders
    
    Args:
        stego_image: Encoded image [B, C, H, W]
        cover_image: Original cover image [B, C, H, W]
    
    Returns:
        gradient_loss: Penalty for sharp gradients near edges
    """
    diff = torch.abs(stego_image - cover_image)
    
    # Compute spatial gradients of the difference
    grad_h = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])  # Horizontal gradient
    grad_v = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])  # Vertical gradient
    
    # Focus on edges (first and last few rows/columns)
    edge_width = 3
    
    # Top/bottom edge gradients
    top_grad = grad_h[:, :, :edge_width, :].mean()
    bottom_grad = grad_h[:, :, -edge_width:, :].mean()
    
    # Left/right edge gradients
    left_grad = grad_v[:, :, :, :edge_width].mean()
    right_grad = grad_v[:, :, :, -edge_width:].mean()
    
    gradient_loss = (top_grad + bottom_grad + left_grad + right_grad) / 4.0
    
    return gradient_loss


def combined_edge_loss(stego_image, cover_image, edge_width=2, 
                       alpha=1.0, beta=0.5):
    """
    Combined edge loss: Edge penalty + gradient penalty
    
    Args:
        stego_image: Encoded image [B, C, H, W]
        cover_image: Original cover image [B, C, H, W]
        edge_width: Number of pixels to consider as edge
        alpha: Weight for edge penalty (default: 1.0)
        beta: Weight for gradient penalty (default: 0.5)
    
    Returns:
        total_edge_loss: Combined edge and gradient penalty
    """
    edge_loss = edge_penalty_loss_simple(stego_image, cover_image, edge_width)
    grad_loss = edge_gradient_penalty(stego_image, cover_image)
    
    return alpha * edge_loss + beta * grad_loss


# ==================== Edge Penalty Loss Class ====================

class EdgePenaltyLoss(nn.Module):
    """
    Edge penalty loss as a PyTorch module
    Combines edge pixel penalty with gradient penalty
    """
    def __init__(self, edge_width=2, alpha=1.0, beta=0.5, loss_type='combined'):
        """
        Args:
            edge_width: Number of pixels to consider as edge (default: 2)
            alpha: Weight for edge pixel penalty (default: 1.0)
            beta: Weight for gradient penalty (default: 0.5)
            loss_type: Type of edge loss
                - 'simple': Simple edge penalty (L1)
                - 'weighted': Weighted edge penalty (outer pixels penalized more)
                - 'l2': L2-based edge penalty
                - 'gradient': Only gradient penalty
                - 'combined': Edge + gradient penalty (recommended)
        """
        super(EdgePenaltyLoss, self).__init__()
        self.edge_width = edge_width
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
    
    def edge_penalty_simple(self, stego_image, cover_image):
        """Simple L1 edge penalty"""
        diff = torch.abs(stego_image - cover_image)
        
        top_edge = diff[:, :, :self.edge_width, :]
        bottom_edge = diff[:, :, -self.edge_width:, :]
        left_edge = diff[:, :, :, :self.edge_width]
        right_edge = diff[:, :, :, -self.edge_width:]
        
        edge_loss = (top_edge.mean() + bottom_edge.mean() + 
                     left_edge.mean() + right_edge.mean()) / 4.0
        
        return edge_loss
    
    def edge_penalty_weighted(self, stego_image, cover_image):
        """Weighted edge penalty - outer pixels penalized more"""
        diff = torch.abs(stego_image - cover_image)
        
        weights = torch.arange(self.edge_width, 0, -1, dtype=torch.float32, device=diff.device)
        weights = weights / weights.sum()
        
        total_loss = 0.0
        
        # Top edge
        for i in range(self.edge_width):
            total_loss += weights[i] * diff[:, :, i, :].mean()
        
        # Bottom edge
        for i in range(self.edge_width):
            total_loss += weights[i] * diff[:, :, -(i+1), :].mean()
        
        # Left edge
        for i in range(self.edge_width):
            total_loss += weights[i] * diff[:, :, :, i].mean()
        
        # Right edge
        for i in range(self.edge_width):
            total_loss += weights[i] * diff[:, :, :, -(i+1)].mean()
        
        return total_loss / 4.0
    
    def edge_penalty_l2(self, stego_image, cover_image):
        """L2-based edge penalty"""
        diff_squared = (stego_image - cover_image) ** 2
        
        top_edge = diff_squared[:, :, :self.edge_width, :]
        bottom_edge = diff_squared[:, :, -self.edge_width:, :]
        left_edge = diff_squared[:, :, :, :self.edge_width]
        right_edge = diff_squared[:, :, :, -self.edge_width:]
        
        edge_loss = (top_edge.mean() + bottom_edge.mean() + 
                     left_edge.mean() + right_edge.mean()) / 4.0
        
        return edge_loss
    
    def gradient_penalty(self, stego_image, cover_image):
        """Gradient-based penalty for sharp changes near edges"""
        diff = torch.abs(stego_image - cover_image)
        
        # Compute spatial gradients
        grad_h = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
        grad_v = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
        
        edge_grad_width = min(3, self.edge_width)
        
        # Edge gradients
        top_grad = grad_h[:, :, :edge_grad_width, :].mean()
        bottom_grad = grad_h[:, :, -edge_grad_width:, :].mean()
        left_grad = grad_v[:, :, :, :edge_grad_width].mean()
        right_grad = grad_v[:, :, :, -edge_grad_width:].mean()
        
        gradient_loss = (top_grad + bottom_grad + left_grad + right_grad) / 4.0
        
        return gradient_loss
    
    def forward(self, stego_image, cover_image):
        """
        Compute edge penalty loss
        
        Args:
            stego_image: Encoded image [B, C, H, W]
            cover_image: Original cover image [B, C, H, W]
        
        Returns:
            loss: Edge penalty loss
        """
        if self.loss_type == 'simple':
            return self.alpha * self.edge_penalty_simple(stego_image, cover_image)
        
        elif self.loss_type == 'weighted':
            return self.alpha * self.edge_penalty_weighted(stego_image, cover_image)
        
        elif self.loss_type == 'l2':
            return self.alpha * self.edge_penalty_l2(stego_image, cover_image)
        
        elif self.loss_type == 'gradient':
            return self.beta * self.gradient_penalty(stego_image, cover_image)
        
        elif self.loss_type == 'combined':
            edge_loss = self.edge_penalty_simple(stego_image, cover_image)
            grad_loss = self.gradient_penalty(stego_image, cover_image)
            return self.alpha * edge_loss + self.beta * grad_loss
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
