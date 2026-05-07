"""
Evaluation metrics for implicit constraint learning.
"""

import numpy as np
import torch

def normalized_distance(model, X, grad_floor=1e-6, batch=4096):
    """
    Compute normalized distance to the zero level set for each point in X.
    Args:
        model: Trained model
        X: Input points (Tensor)
        grad_floor: Small value to avoid division by zero
        batch: Batch size for computation
    Returns:
        np.ndarray: Normalized distances
    """
    out = []
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = X[i:i+batch].detach().clone().requires_grad_(True)
        hb = model(xb)
        grad = torch.autograd.grad(hb.sum(), xb, create_graph=False)[0]
        dn = hb.abs() / (grad.norm(dim=1) + grad_floor)
        out.append(dn.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

def data_sat_score(model, pos_tensor, tau=0.03):
    """
    Compute the data satisfaction score (fraction of points within tau of the zero set).
    Args:
        model: Trained model
        pos_tensor: Positive samples (Tensor)
        tau: Distance threshold
    Returns:
        float: Satisfaction score
        np.ndarray: Normalized distances
    """
    dhat = normalized_distance(model, pos_tensor)
    return float((dhat <= tau).mean()), dhat

def chamfer_symmetric(A, B):
    """
    Symmetric Chamfer distance between two point clouds A and B.
    Args:
        A, B: np.ndarray of shape (N, 2)
    Returns:
        float: Symmetric Chamfer distance
    """
    if A.shape[0]==0 or B.shape[0]==0:
        return np.inf
    d2 = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
    da = np.sqrt(d2.min(axis=1)).mean()
    db = np.sqrt(d2.min(axis=0)).mean()
    return (da + db) * 0.5

def geometry_chamfer_score(model, gt_curve_pts, sigma=0.06):
    """
    Compute geometry score using Chamfer distance between model and GT curve.
    FIXED VERSION - matches v1.py exactly
    Args:
        model: Trained model
        gt_curve_pts: Ground truth curve points (np.ndarray)
        sigma: Score scale (default 0.06)
    Returns:
        float: Score
        float: Chamfer distance
    """
    pred_pts = grid_zero_contour_points(model, grid_lim=1.8, N=240)
    cd = chamfer_symmetric(pred_pts, gt_curve_pts)
    if not np.isfinite(cd): 
        return 0.0, cd
    score = float(np.exp(-cd / sigma))
    return score, cd

def grid_zero_contour_points(model, grid_lim=1.8, N=240):
    """
    Extracts the (x, y) points of the model's zero-level set (contour where model(x, y) = 0).
    FIXED: Use same defaults as v1.py
    Args:
        model: Trained model
        grid_lim: Range for the grid (default 1.8 - MATCHES v1.py)
        N: Grid resolution (default 240 - MATCHES v1.py)
    Returns:
        np.ndarray: Array of (x, y) points on the zero-level set
    """
    import matplotlib.pyplot as plt
    xs = np.linspace(-grid_lim, grid_lim, N)
    ys = np.linspace(-grid_lim, grid_lim, N)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    import torch
    with torch.no_grad():
        ZZ = model(torch.from_numpy(P)).cpu().numpy().reshape(N, N)
    fig, ax = plt.subplots()
    cs = ax.contour(XX, YY, ZZ, levels=[0.0])
    plt.close(fig)
    seglists = getattr(cs, "allsegs", None)
    if not seglists or len(seglists) == 0:
        return np.zeros((0, 2))
    level_segs = seglists[0]  
    if len(level_segs) == 0:
        return np.zeros((0, 2))
    return np.concatenate(level_segs, axis=0)




def extract_surface_points(model, level=0.0, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
                           resolution=50, batch=8192, n_samples=5000):
    """
    Extract surface points from model using marching cubes, then uniformly sample.
    
    
    Args:
        model: Trained model
        level: Isosurface level to extract
        bounds: Volume bounds
        resolution: Grid resolution for marching cubes
        batch: Batch size for model evaluation
        n_samples: Number of points to sample uniformly on surface
        
    Returns:
        np.ndarray: (N, 3) array of points on the surface, or empty array if extraction fails
    """
    from skimage.measure import marching_cubes
    
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    z = np.linspace(zmin, zmax, resolution)
    
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    
    # Evaluate model on grid
    grid_tensor = torch.from_numpy(grid_points).float()
    device = next(model.parameters()).device
    grid_tensor = grid_tensor.to(device)
    
    model.eval()
    model_out = []
    n = grid_tensor.shape[0]
    
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = grid_tensor[i:i+batch]
            hb = model(xb)
            model_out.append(hb.cpu().numpy())
    
    model_values = np.concatenate(model_out, axis=0).reshape(resolution, resolution, resolution)
    
    # Extract isosurface with marching cubes
    try:
        if not (np.any(model_values <= level) and np.any(model_values >= level)):
            print(f"      [Surface Extract] WARNING: No level crossing at {level:.3f}")
            return np.empty((0, 3))
        
        dx = (xmax - xmin) / (resolution - 1)
        dy = (ymax - ymin) / (resolution - 1)
        dz = (zmax - zmin) / (resolution - 1)
        
        verts, faces, _, _ = marching_cubes(model_values, level=level, spacing=(dx, dy, dz))
        verts[:, 0] += xmin
        verts[:, 1] += ymin
        verts[:, 2] += zmin
        
        # Uniformly sample points on the triangular mesh
        if len(faces) == 0:
            print(f"      [Surface Extract] WARNING: No faces extracted")
            return np.empty((0, 3))
        
        # Compute triangle areas for weighted sampling
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        total_area = np.sum(areas)
        
        if total_area < 1e-12:
            print(f"      [Surface Extract] WARNING: Surface has zero area")
            return np.empty((0, 3))
        
        # Sample triangles proportional to their area
        probs = areas / total_area
        triangle_indices = np.random.choice(len(faces), size=n_samples, p=probs)
        
        # Sample uniformly within each selected triangle (barycentric coordinates)
        r1 = np.random.random(n_samples)
        r2 = np.random.random(n_samples)
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        
        sampled_points = (
            u[:, None] * verts[faces[triangle_indices, 0]] +
            v[:, None] * verts[faces[triangle_indices, 1]] +
            w[:, None] * verts[faces[triangle_indices, 2]]
        )
        
        print(f"      [Surface Extract] {len(verts)} verts, {len(faces)} faces, "
              f"sampled {len(sampled_points)} pts (area={total_area:.2f})")
        
        return sampled_points
        
    except Exception as e:
        print(f"      [Surface Extract] ERROR: {e}")
        return np.empty((0, 3))


def chamfer_distance_symmetric(points_A, points_B):
    """
    Symmetric Chamfer distance between two point clouds.
    
    Chamfer(A, B) = mean(min_dist(A→B)) + mean(min_dist(B→A))
    
    THE ONLY METRIC YOU NEED to compare implicit surfaces.
    
    Args:
        points_A: (N, 3) numpy array
        points_B: (M, 3) numpy array
        
    Returns:
        float: Symmetric Chamfer distance (lower = better surface)
    """
    if len(points_A) == 0 or len(points_B) == 0:
        return np.inf
    
    # A → B: for each point in A, find nearest point in B
    diff_AB = points_A[:, None, :] - points_B[None, :, :]  # (N, M, 3)
    dist_AB = np.linalg.norm(diff_AB, axis=2)  # (N, M)
    min_dist_AB = np.min(dist_AB, axis=1)  # (N,)
    chamfer_AB = np.mean(min_dist_AB)
    
    # B → A: for each point in B, find nearest point in A
    diff_BA = points_B[:, None, :] - points_A[None, :, :]  # (M, N, 3)
    dist_BA = np.linalg.norm(diff_BA, axis=2)  # (M, N)
    min_dist_BA = np.min(dist_BA, axis=1)  # (M,)
    chamfer_BA = np.mean(min_dist_BA)
    
    # Symmetric Chamfer
    chamfer_sym = chamfer_AB + chamfer_BA
    
    return float(chamfer_sym)


def compute_chamfer_distance(model, gt_samples_fn, level=0.0, 
                             bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
                             resolution=50, n_samples=5000, seed=42):
    """
    Compute Chamfer distance between model's learned surface and GT surface.
    
    Chamfer Distance (CD) compares two point clouds WITHOUT requiring:
    - Same number of points
    - Point-to-point correspondence
    - Same distribution
    
    CD(A, G) = (1/|A|) Σ min||a-g||² + (1/|G|) Σ min||g-a||²
              └─ Precision ─┘         └─ Recall ──┘
    
    The metric is already normalized by set sizes, so NO need to force equal counts.
    
    Lower Chamfer = Better surface approximation.
    
    Args:
        model: Trained model
        gt_samples_fn: Function that generates GT surface samples (e.g., gt_samples from datasets3D)
        level: Isosurface level for model (default 0.0)
        bounds: Volume bounds
        resolution: Grid resolution for marching cubes
        n_samples: Target number of points to sample (actual counts may vary)
        seed: Random seed for reproducible sampling (default 42)
        
    Returns:
        float: Chamfer distance (lower is better)
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Extract GT surface points directly from GT sampling function
    gt_points = gt_samples_fn(n_samples)
    
    # Extract predicted surface using marching cubes + uniform sampling
    pred_points = extract_surface_points(
        model, level=level, bounds=bounds, resolution=resolution, n_samples=n_samples
    )
    
    if len(gt_points) == 0 or len(pred_points) == 0:
        print(f"      [Chamfer] ERROR: Could not extract surfaces")
        return np.inf
    
    # Compute Chamfer distance
    # No need to force equal counts - CD is normalized by definition!
    chamfer = chamfer_distance_symmetric(pred_points, gt_points)
    
    print(f"      [Chamfer] {chamfer:.6f} (GT={len(gt_points)} pts, Pred={len(pred_points)} pts)")
    
    return float(chamfer)
