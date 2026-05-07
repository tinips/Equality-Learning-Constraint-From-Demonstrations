"""
Reference Paper Losses (ECoMaNN-style Siamese Losses)

Faithful implementation of the Siamese losses from:
"Learning Equality Constraints for Motion Planning on Manifolds" (arXiv:2009.11852)

Loss components:
- L_norm: ||h(q̌)||₂ = i*ε
- L_reflection: ||h(q + iεu) + h(q - iεu)||²  
- L_fraction: normalized outputs match for fractional points
- L_similar: aligned augmented points have same h value
- L_subspace: Jacobian-PCA alignment

OSA (Orthogonal Subspace Augmentation) via data augmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


def norm_loss(model: torch.nn.Module, 
              aug_data: torch.Tensor, 
              aug_levels: torch.Tensor, 
              epsilon: float) -> torch.Tensor:
    """
    L_norm (Paper Eq.): ||h(q̌)||₂ = i*ε
    
    The magnitude of h at augmented point q̌ = q + i*ε*u should equal i*epsilon.
    
    Args:
        model: Neural network model h(x)
        aug_data: Augmented samples q̌ (M, d)
        aug_levels: Augmentation level i for each sample (M,)
        epsilon: Base augmentation distance ε
        
    Returns:
        loss: Norm loss value
    """
    f_aug = model(aug_data)  # (M, l)
    target_norms = aug_levels.float() * epsilon  # (M,)
    
    # For single output (l=1)
    if f_aug.dim() == 1 or f_aug.shape[-1] == 1:
        predicted_norms = f_aug.abs().squeeze(-1)
    else:
        predicted_norms = f_aug.norm(dim=1)
    
    loss = ((predicted_norms - target_norms) ** 2).mean()
    return loss


def reflection_loss(model: torch.nn.Module, 
                    x_pos: torch.Tensor, 
                    normals: torch.Tensor,
                    epsilon: float = 1.0) -> torch.Tensor:
    """
    L_reflection (Paper Eq.): ||h(q + iεu) + h(q - iεu)||²
    
    Symmetric points along the normal direction should have opposite h values.
    h(q + εu) = -h(q - εu), so their sum should be zero.
    
    Args:
        model: Neural network model h(x)
        x_pos: On-manifold samples q (N, d)
        normals: Normal vectors at each point (N, d)
        epsilon: Distance along normal (default: 1.0)
        
    Returns:
        loss: Reflection loss value
    """
    # Normalize normals
    normals_normalized = F.normalize(normals, dim=1, eps=1e-8)
    
    # Create symmetric pairs
    x_plus = x_pos + epsilon * normals_normalized   # q + εu
    x_minus = x_pos - epsilon * normals_normalized  # q - εu
    
    # Evaluate h at both points
    f_plus = model(x_plus)   # h(q + εu)
    f_minus = model(x_minus) # h(q - εu)
    
    # Paper: h(q+εu) + h(q-εu) should be 0
    loss = ((f_plus + f_minus) ** 2).mean()
    
    return loss


def fraction_loss(model: torch.nn.Module, 
                  x_pos: torch.Tensor, 
                  normals: torch.Tensor,
                  epsilon: float = 1.0,
                  fractions: Tuple[float, ...] = (0.25, 0.5, 0.75)) -> torch.Tensor:
    """
    L_fraction (Paper Eq.): normalized outputs match for fractional points.
    
    h(q+εu)/||h(q+εu)|| = h(q+(a/b)εu)/||h(q+(a/b)εu)||
    
    Points at different fractions along the normal should have the same
    normalized output direction.
    
    Args:
        model: Neural network model h(x)
        x_pos: On-manifold samples q (N, d)
        normals: Normal vectors at each point (N, d)
        epsilon: Full distance along normal
        fractions: Fraction values a/b to compare (default: 0.25, 0.5, 0.75)
        
    Returns:
        loss: Fraction loss value
    """
    # Normalize normals
    normals_normalized = F.normalize(normals, dim=1, eps=1e-8)
    
    # Reference point at full epsilon
    x_full = x_pos + epsilon * normals_normalized
    f_full = model(x_full)
    f_full_normalized = F.normalize(f_full, dim=-1, eps=1e-8)
    
    # Compare with fractional points
    frac_losses = []
    for frac in fractions:
        x_frac = x_pos + frac * epsilon * normals_normalized
        f_frac = model(x_frac)
        f_frac_normalized = F.normalize(f_frac, dim=-1, eps=1e-8)
        
        # Normalized outputs should match
        frac_loss = ((f_full_normalized - f_frac_normalized) ** 2).mean()
        frac_losses.append(frac_loss)
    
    loss = torch.stack(frac_losses).mean()
    return loss


def similarity_loss(model: torch.nn.Module, 
                    x_aug_a: torch.Tensor, 
                    x_aug_c: torch.Tensor) -> torch.Tensor:
    """
    L_similar (Paper Eq.): ||h(q_a + iεu_a) - h(q_c + iεu_c)||²
    
    Two aligned augmented points in the same level set should have the same h value.
    This requires OSA (Orthogonal Subspace Alignment) preprocessing.
    
    Args:
        model: Neural network model h(x)
        x_aug_a: First set of aligned augmented points (N, d)
        x_aug_c: Second set of aligned augmented points (N, d)
        
    Returns:
        loss: Similarity loss value
    """
    f_a = model(x_aug_a)
    f_c = model(x_aug_c)
    
    loss = ((f_a - f_c) ** 2).mean()
    return loss


def subspace_alignment_loss(model: torch.nn.Module, 
                            x_pos: torch.Tensor, 
                            pca_normal_basis: torch.Tensor,
                            output_dim: int = 1) -> torch.Tensor:
    """
    L_subspace (Paper Eq.): Align Jacobian row space with PCA normal space.
    
    ||V_N V_N^T E_N||² + ||E_N E_N^T V_N||²
    
    Where V_N is PCA normal basis, E_N is Jacobian null space basis.
    For l=1, simplifies to gradient-normal alignment.
    
    Args:
        model: Neural network model h(x)
        x_pos: On-manifold samples q (N, d)
        pca_normal_basis: Normal space basis from PCA (N, l, d) or (N, d) for l=1
        output_dim: Output dimension l of the model
        
    Returns:
        loss: Subspace alignment loss value
    """
    N = x_pos.shape[0]
    
    # Compute Jacobian (gradients)
    x_grad = x_pos.clone().detach().requires_grad_(True)
    f = model(x_grad)
    
    if output_dim == 1:
        # Simple case: gradient should align with PCA normal
        grads = torch.autograd.grad(f.sum(), x_grad, create_graph=True)[0]
        grads_normalized = F.normalize(grads, dim=1)
        
        # Normalize PCA normals
        if pca_normal_basis.dim() == 3:
            pca_normals = pca_normal_basis[:, 0, :]  # (N, d)
        else:
            pca_normals = pca_normal_basis
        pca_normals_normalized = F.normalize(pca_normals, dim=1)
        
        # Alignment: 1 - cos²(angle) should be 0
        cos_angle = (grads_normalized * pca_normals_normalized).sum(dim=1)
        loss = (1.0 - cos_angle.abs() ** 2).mean()
    else:
        # General case for l > 1
        # Compute full Jacobian matrix (N, l, d)
        J = torch.stack([
            torch.autograd.grad(f[:, i].sum(), x_grad, 
                              retain_graph=True, create_graph=True)[0]
            for i in range(output_dim)
        ], dim=1)
        
        V_N = pca_normal_basis  # (N, l, d)
        E_N = F.normalize(J, dim=2)  # Normalize each row
        
        # Projection errors
        loss_terms = []
        for i in range(N):
            V = V_N[i]  # (l, d)
            E = E_N[i]  # (l, d)
            
            # ||V V^T E - E||² (E should be in span of V)
            proj_E = V @ (V.T @ E)
            error1 = (proj_E - E).pow(2).sum()
            
            # ||E E^T V - V||² (V should be in span of E)
            proj_V = E @ (E.T @ V)
            error2 = (proj_V - V).pow(2).sum()
            
            loss_terms.append(error1 + error2)
        
        loss = torch.stack(loss_terms).mean()
    
    return loss


def ecomann_loss(model: torch.nn.Module, 
                 x_pos: torch.Tensor, 
                 aug_data: torch.Tensor,
                 aug_levels: torch.Tensor,
                 epsilon: float,
                 pca_normals: torch.Tensor,
                 lambda_norm: float = 1.0,
                 lambda_reflection: float = 1.0, 
                 lambda_fraction: float = 1.0,
                 lambda_subspace: float = 1.0,
                 x_similar_a: Optional[torch.Tensor] = None,
                 x_similar_c: Optional[torch.Tensor] = None,
                 lambda_similar: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined ECoMaNN Loss (exact paper formulation).
    
    Combines all paper loss components:
    - L_norm: Augmented point magnitude
    - L_reflection: Symmetric point antisymmetry
    - L_fraction: Normalized output consistency
    - L_subspace: Jacobian-PCA alignment
    - L_similar: Aligned point similarity (optional, requires OSA)
    
    Args:
        model: Neural network model h(x)
        x_pos: On-manifold samples (N, d)
        aug_data: Augmented samples (M, d)
        aug_levels: Augmentation levels (M,)
        epsilon: Base augmentation distance
        pca_normals: Normal vectors (N, d)
        lambda_*: Loss weights
        x_similar_a, x_similar_c: Aligned augmented pairs for similarity loss
        
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Individual loss components
    """
    # Compute individual losses
    loss_norm = norm_loss(model, aug_data, aug_levels, epsilon)
    loss_refl = reflection_loss(model, x_pos, pca_normals, epsilon)
    loss_frac = fraction_loss(model, x_pos, pca_normals, epsilon)
    loss_sub = subspace_alignment_loss(model, x_pos, pca_normals)
    
    # Combine losses
    total_loss = (lambda_norm * loss_norm + 
                  lambda_reflection * loss_refl + 
                  lambda_fraction * loss_frac +
                  lambda_subspace * loss_sub)
    
    loss_dict = {
        'norm': loss_norm.item(),
        'reflection': loss_refl.item(),
        'fraction': loss_frac.item(),
        'subspace': loss_sub.item(),
    }
    
    # Add similarity loss if aligned pairs provided
    if x_similar_a is not None and x_similar_c is not None:
        loss_sim = similarity_loss(model, x_similar_a, x_similar_c)
        total_loss = total_loss + lambda_similar * loss_sim
        loss_dict['similarity'] = loss_sim.item()
    
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


# =============================================================================
# OSA (Orthogonal Subspace Alignment) - Paper Algorithm
# =============================================================================

def compute_osa_alignment(pos_samples: torch.Tensor,
                          pca_normal_bases: torch.Tensor,
                          k_neighbors: int = 10) -> torch.Tensor:
    """
    OSA (Orthogonal Subspace Alignment) preprocessing.
    
    Aligns normal spaces of neighboring on-manifold points globally using:
    1. K-NN graph construction
    2. MST (Minimum Spanning Tree) → DAG
    3. Pairwise rotation optimization
    4. Global alignment propagation
    
    Args:
        pos_samples: On-manifold points (N, d)
        pca_normal_bases: Local normal bases (N, l, d)
        k_neighbors: Number of neighbors for graph
        
    Returns:
        aligned_bases: Globally aligned normal bases (N, l, d)
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree
    import numpy as np
    
    N = pos_samples.shape[0]
    device = pos_samples.device
    
    # Convert to numpy for graph operations
    pos_np = pos_samples.detach().cpu().numpy()
    bases_np = pca_normal_bases.detach().cpu().numpy()
    
    # Build KNN graph
    tree = cKDTree(pos_np)
    distances, indices = tree.query(pos_np, k=k_neighbors + 1)
    
    # Create sparse distance matrix for MST
    from scipy.sparse import lil_matrix
    dist_matrix = lil_matrix((N, N))
    for i in range(N):
        for j, d in zip(indices[i, 1:], distances[i, 1:]):
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # Compute MST
    mst = minimum_spanning_tree(dist_matrix.tocsr())
    mst_edges = list(zip(*mst.nonzero()))
    
    # BFS from root to align all bases
    aligned = np.zeros(N, dtype=bool)
    aligned_bases = bases_np.copy()
    
    # Start from first node
    queue = [0]
    aligned[0] = True
    
    while queue:
        current = queue.pop(0)
        
        # Find neighbors in MST
        for i, j in mst_edges:
            if i == current and not aligned[j]:
                # Align j's basis to i's basis
                aligned_bases[j] = _align_bases(aligned_bases[i], bases_np[j])
                aligned[j] = True
                queue.append(j)
            elif j == current and not aligned[i]:
                aligned_bases[i] = _align_bases(aligned_bases[j], bases_np[i])
                aligned[i] = True
                queue.append(i)
    
    return torch.from_numpy(aligned_bases).float().to(device)


def _align_bases(V_ref: 'np.ndarray', V_target: 'np.ndarray') -> 'np.ndarray':
    """
    Align target basis to reference basis by finding optimal rotation.
    
    Handles sign ambiguity by testing all sign combinations.
    
    Args:
        V_ref: Reference normal basis (l, d)
        V_target: Target normal basis to align (l, d)
        
    Returns:
        V_aligned: Aligned target basis (l, d)
    """
    import numpy as np
    
    l = V_ref.shape[0]
    
    # For l=1, just match signs
    if l == 1:
        if np.dot(V_ref.flatten(), V_target.flatten()) < 0:
            return -V_target
        return V_target
    
    # For l>1, find rotation that minimizes ||I - (V_ref @ R)^T @ V_target||
    # Test sign combinations
    best_error = float('inf')
    best_aligned = V_target
    
    for signs in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        V_test = V_target.copy()
        for i, s in enumerate(signs[:min(l, 2)]):
            V_test[i] *= s
        
        # Compute alignment error
        error = np.linalg.norm(np.eye(l) - V_ref @ V_test.T)
        if error < best_error:
            best_error = error
            best_aligned = V_test
    
    return best_aligned


# =============================================================================
# TANGENT SPACE SAMPLING FOR OSA AUGMENTATION
# =============================================================================

def sample_tangent_space_points(x_pos: torch.Tensor, 
                                 normals: torch.Tensor, 
                                 epsilon: float = 0.1) -> torch.Tensor:
    """
    Sample points in the tangent space for similarity loss.
    
    Creates orthogonal basis in tangent space using Gram-Schmidt
    and samples points nearby in that subspace.
    
    Args:
        x_pos: On-manifold samples (N, d)
        normals: Normal vectors at each point (N, d)
        epsilon: Sampling radius in tangent space
        
    Returns:
        x_similar: Points sampled in tangent space (N, d)
    """
    N, d = x_pos.shape
    device = x_pos.device
    
    # Normalize normals
    normals_normalized = F.normalize(normals, dim=1, eps=1e-8)
    
    # Create orthogonal basis in tangent space via Gram-Schmidt
    # Random vector for starting point
    random_vec = torch.randn(N, d, device=device)
    
    # First tangent vector: project out normal component
    dot = (random_vec * normals_normalized).sum(dim=1, keepdim=True)
    tangent1 = random_vec - dot * normals_normalized
    tangent1 = F.normalize(tangent1, dim=1, eps=1e-8)
    
    # Second tangent vector: cross product (for 3D)
    if d == 3:
        tangent2 = torch.cross(normals_normalized, tangent1, dim=1)
        tangent2 = F.normalize(tangent2, dim=1, eps=1e-8)
        
        # Sample random coefficients
        alpha1 = torch.randn(N, 1, device=device) * epsilon
        alpha2 = torch.randn(N, 1, device=device) * epsilon
        
        x_similar = x_pos + alpha1 * tangent1 + alpha2 * tangent2
    else:
        # For higher dimensions, just use one tangent direction
        alpha1 = torch.randn(N, 1, device=device) * epsilon
        x_similar = x_pos + alpha1 * tangent1
    
    return x_similar
