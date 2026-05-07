import torch
import torch.nn.functional as F

def pos_loss(model, xb_p):
	hp = model(xb_p)
	return hp.abs().mean()

def pos_neg_loss(model, xb_p, xb_n, margin, alpha):
	hp = model(xb_p)
	hn = model(xb_n)
	return hp.abs().mean() + alpha * F.softplus(margin - hn.abs()).mean()

def knn_weighted_neg_loss(model, xb_n, xb_p, margin, alpha):
    k = 3
    d2 = ((xb_n[:, None, :] - xb_p[None, :, :]) ** 2).sum(-1)  
    dists, _ = d2.topk(k, dim=1, largest=False)  
    mean_knn_dist = dists.sqrt().mean(dim=1)
    
    adaptive_margin = mean_knn_dist.detach()
    
    hn = model(xb_n)
    loss = F.softplus(adaptive_margin - hn.abs()).mean()
    return alpha * loss

def neural_pull_loss(model, xb_n, xb_p, return_projection_info=False):

    all_points = torch.cat([xb_p, xb_n], dim=0)
    all_points_grad = all_points.detach().clone().requires_grad_(True)
    
    f_vals = model(all_points_grad)
    f_vals = torch.nan_to_num(f_vals, nan=0.0, posinf=1e6, neginf=-1e6)
    
    grads = torch.autograd.grad(f_vals.sum(), all_points_grad, create_graph=True)[0]
    grads = torch.nan_to_num(grads, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Clamp grad norms to avoid division by very small values
    grad_norms = grads.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # Project points to constraint surface: x' = x - η * f(x) * ∇f(x) / |∇f(x)|
    projected_points = all_points_grad - f_vals.unsqueeze(1) * grads / grad_norms
    projected_points = torch.nan_to_num(projected_points, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Find k-NN of projected points in positive dataset
    # Safe k: ensure we don't request more neighbors than available
    n_pos = xb_p.shape[0]
    k = min(3, n_pos)


    # Create a detached copy of xb_p for distance computation to avoid gradient issues
    xb_p_detached = xb_p.detach()
    
    # Compute distances from projected points to all positives
    # Shape: (n_all_points, n_positives)
    dists_to_pos = ((projected_points[:, None, :] - xb_p_detached[None, :, :]) ** 2).sum(-1)
    dists_to_pos = dists_to_pos.clamp(min=0.0)  # ensure non-negative before sqrt
    
    # Find k nearest neighbors for each projected point
    knn_dists, _ = dists_to_pos.topk(k, dim=1, largest=False)
    
    # Neural Pull loss: minimize average distance to k-NN
    # Add small eps before sqrt to avoid sqrt(0) = NaN gradient
    pull_loss = (knn_dists + 1e-8).sqrt().mean()
    
    # Add positive constraint: force f(x_pos) = 0
    pos_constraint = model(xb_p).abs().mean()
    
    # Combine both terms. Apply optional down-weighting for parametric quadratics
    total_loss = pos_constraint + pull_loss

    if return_projection_info:
        # Store projection information for visualization
        with torch.no_grad():
            proj_f_vals = model(projected_points.detach())
        
        projection_info = {
            'original_points': all_points.detach(),
            'projected_points': projected_points.detach(), 
            'original_f_vals': f_vals.detach(),
            'projected_f_vals': proj_f_vals,
            'n_pos': xb_p.shape[0],
            'n_neg': xb_n.shape[0]
        }
        return total_loss, projection_info
    
    return total_loss