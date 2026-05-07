"""
Dataset generation for imitation controller training.

This module creates training data for an imitation learning controller by:
1. Taking a learned implicit surface model
2. Generating diverse trajectories on the learned surface
3. Building (current_pos, goal) -> next_pos training pairs
"""

import numpy as np
import torch
from .datasets3D import make_constraint_3d, generate_surface_trajectory

from src.utils.seed import set_seed
from src.data.datasets3D import make_constraint_3d
def h_learned(model, points):
    """Evaluate learned implicit function h(x) for array of points.
    
    Args:
        model: Trained implicit surface model (nn.Module)
        points: (N, 3) array of 3D points
        
    Returns:
        (N,) array of h values
    """
    points = np.atleast_2d(points)
    with torch.no_grad():
        pts_t = torch.from_numpy(points.astype(np.float32))
        out = model(pts_t).squeeze(-1).cpu().numpy()
    if out.ndim == 0:
        out = np.array([out])
    return out


def grad_learned(model, points):
    """Compute gradient (normal) of learned surface at points using autograd.
    
    Args:
        model: Trained implicit surface model (nn.Module)
        points: (N, 3) array of 3D points
        
    Returns:
        (N, 3) array of normalized gradients
    """
    points = np.atleast_2d(points)
    grads = []
    for i in range(len(points)):
        p = torch.tensor(points[i:i+1], dtype=torch.float32, requires_grad=True)
        with torch.enable_grad():
            h_val = model(p).squeeze()
            g = torch.autograd.grad(outputs=h_val, inputs=p, create_graph=False)[0]
        grad_np = g.detach().numpy()[0]
        # Normalize
        grad_np = grad_np / (np.linalg.norm(grad_np) + 1e-9)
        grads.append(grad_np)
    return np.array(grads)


def trajectories_to_states_targets(trajectories):
        """
        Donades trajectòries de test, retorna arrays (states, targets).
        Accepta cada entrada com:
          - una seqència d'punts (N,3) o
          - un tuple (start_pt, goal_pt, real_traj)
        State format: [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        Target format: delta = next_pos - current (3,)
        """
        states = []
        targets = []
        for item in trajectories:
            
            seq = np.asarray(item)
            goal = seq[-1].reshape(-1)

            # Build samples
            for j in range(len(seq) - 1):
                current = np.asarray(seq[j]).reshape(-1)
                next_pos = np.asarray(seq[j + 1]).reshape(-1)
                state = np.concatenate([current, goal])
                delta = next_pos - current
                states.append(state)
                targets.append(delta)

        return np.array(states), np.array(targets)