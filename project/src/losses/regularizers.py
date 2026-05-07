
import torch
from torch.func import grad, vmap


def param_l2(model):
    return sum((p**2).sum() for p in model.parameters())


def eikonal_penalty(model, xs, target_norm=1.0):
    # Define gradient function for a single point
    def single_grad(model, x):
        """Compute ∇h(x) for a single input point."""
        return grad(model)(x)
    
    # Vectorize over batch dimension: compute gradients for all points
    grad_batch = vmap(single_grad, in_dims=(None, 0))(model, xs)
    
    # Compute Eikonal penalty: ||∇h|| should equal target_norm
    grad_norms = grad_batch.norm(dim=1)
    return ((grad_norms - target_norm)**2).mean()
