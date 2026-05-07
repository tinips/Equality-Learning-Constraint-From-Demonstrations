from src.data.datasets3D import *

from src.utils.viz3D import *
import numpy as np

def pca_planarity_test(pos_np):
        
    centeroid = pos_np.mean(axis=0)
    posc = pos_np - centeroid
    cov = (posc.T @ posc) / posc.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    normal = eigvecs[:, -1]
    normal = normal / (np.linalg.norm(normal))
    r = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else 0.0

  
    if r < 0.01:
        
        return True
    else:
        
        return False
    

def is_near_spherical(X, threshold = 0.01):
 
    X = np.asarray(X, dtype=float)
    c = X.mean(axis=0)                      
    r = np.linalg.norm(X - c, axis=1)        
    ratio = float(np.var(r) / (np.mean(r) ** 2)) if np.mean(r) > 0 else np.inf
    return ratio < threshold


def is_locally_1d(Y: np.ndarray, k = 20, samples = 200, ratio_thresh = 0.15, vote_frac = 0.6):
    """Return True if pointcloud Y is locally 1D (curve-like) using local PCA.

    Parameters
    - Y: (N,3) point cloud
    - k: local neighborhood size (number of neighbors)
    - samples: number of center points to sample for voting
    - ratio_thresh: how much variance must NOT be in other directions
    - vote_frac: fraction of sampled neighborhoods that must be 1D
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    if n < k + 1:
        return False
    # sample indices (avoid heavy cost on huge point clouds)
    if n <= samples:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(n, size=samples, replace=False)

    votes = 0
    for i in idx:
        pi = Y[i:i+1]
        # compute squared distances to all points
        d2 = np.sum((Y - pi) ** 2, axis=1)
        # get k+1 (including itself) smallest indices
        nn = np.argsort(d2)[: (k + 1)]
        nbrs = Y[nn]
        # center and covariance
        nbrs_c = nbrs - nbrs.mean(axis=0)
        cov = (nbrs_c.T @ nbrs_c) / max(1, nbrs_c.shape[0])
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        s = np.sum(eigvals)
        if s <= 0:
            continue
        # fraction of variance in the first principal direction
        frac1 = eigvals[0] / s
        # if first eigenvalue explains most variance -> locally 1D
        if frac1 > (1.0 - ratio_thresh):
            votes += 1

    return (votes / len(idx)) > vote_frac

def is_near_cylindrical(X, threshold = 0.01):
    # Quick local-PCA check: if the pointcloud is locally 1D (curve/helix)
    # then we should NOT treat it as a cylindrical surface even if the
    # global radii around some axis look almost constant.

    if is_locally_1d(X):
        return False

    c_xy = X[:, :2].mean(axis=0)

    r = np.linalg.norm(X[:, :2] - c_xy, axis=1)

    ratio = np.var(r) / (np.mean(r) ** 2)

    return ratio < threshold

