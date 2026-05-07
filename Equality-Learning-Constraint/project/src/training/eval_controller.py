"""
Evaluation utilities for imitation controller.
"""

import numpy as np
import torch
from scipy.optimize import minimize


def generate_trajectory_qp(controller, start_pt, goal_pt, h_fn, grad_fn, max_steps=30):

    print('\n========== GENERATING LEARNED TRAJECTORY WITH QP ==========')    
    learned_traj = [start_pt.copy()]
    current = start_pt.copy()
    
    qp_failures = 0
    controller.eval()

    with torch.no_grad():
        for step in range(max_steps - 1):

            # NN prediction
            state = np.concatenate([current, goal_pt]).astype(np.float32)
            delta = controller(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()
            unconstrained_next = current + delta

            # Newton projection with line search
            x = unconstrained_next.copy()
            converged = False
            
            for newton_iter in range(50):
                h_val = h_fn(x)
                if abs(h_val) < 1e-6:
                    converged = True
                    break

                grad = grad_fn(x)
                grad_norm_sq = np.dot(grad, grad)
                if grad_norm_sq < 1e-12:
                    break

                # Newton step direction
                step_dir = -h_val * grad / grad_norm_sq

                alpha = 1.0
                for _ in range(10):
                    new_x = x + alpha * step_dir
                    if abs(h_fn(new_x)) < abs(h_val):   # ensures improvement
                        x = new_x
                        break
                    alpha *= 0.5  # reduce step size if unstable

            if converged:
                next_pos = x
            else:
                qp_failures += 1
                next_pos = unconstrained_next

            learned_traj.append(next_pos)
            current = next_pos

    if qp_failures > 0:
        print(f'  Newton projection failed {qp_failures}/{max_steps-1} times')

    return np.array(learned_traj)


def generate_trajectory(controller, start_pt, goal_pt, max_steps=30):
    """Generate an unconstrained NN rollout (no projection).

    - Uses controller predictions (delta) in closed-loop: x_{t+1} = x_t + delta(x_t, goal)
    - Stops early if the trajectory reaches the goal within `tol`.

    Args:
        controller: nn.Module that maps state (6,) -> delta (3,)
        start_pt: (3,) numpy start point
        goal_pt: (3,) numpy goal point
        max_steps: maximum number of points to generate (including start)
        tol: stopping tolerance on distance to goal (added as kwarg below)
    """
    tol = 1e-3
    learned_traj = [np.asarray(start_pt, dtype=np.float32).copy()]
    current = np.asarray(start_pt, dtype=np.float32).copy()

    controller.eval()

    device = next(controller.parameters()).device if any(p.requires_grad for p in controller.parameters()) else torch.device('cpu')

    with torch.no_grad():
        for step in range(max_steps - 1):
            # build input state [current(3), goal(3)]
            state = np.concatenate([current, np.asarray(goal_pt, dtype=np.float32)])
            state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            delta_t = controller(state_t).squeeze(0).cpu().numpy()
            next_pos = current + delta_t

            learned_traj.append(next_pos.copy())
            current = next_pos.copy()

            # Early stopping if close to goal
            if np.linalg.norm(current - np.asarray(goal_pt, dtype=np.float32)) < tol:
                break

    return np.array(learned_traj)





def evaluate_trajectory_accuracy(real_traj, learned_traj):
    """Calculate accuracy metrics comparing real vs learned trajectory.
    
    Args:
        real_traj: (N, 3) Real/ground truth trajectory
        learned_traj: (M, 3) Learned trajectory from controller
        
    Returns:
        metrics: dict with trajectory comparison metrics
    """
    # Ensure same length for comparison
    min_len = min(len(real_traj), len(learned_traj))
    real_traj = real_traj[:min_len]
    learned_traj = learned_traj[:min_len]
    
    # Point-wise Euclidean distance at each step
    point_distances = np.linalg.norm(real_traj - learned_traj, axis=1)
    
    # Average distance between trajectories
    mean_distance = np.mean(point_distances)
    
    # Maximum deviation
    max_distance = np.max(point_distances)
    
    # Standard deviation of distances
    std_distance = np.std(point_distances)
    
    # Final point error (distance to goal)
    final_error = point_distances[-1]
    
    # Path length ratio (how much longer/shorter is learned path)
    real_path_length = np.sum(np.linalg.norm(np.diff(real_traj, axis=0), axis=1))
    learned_path_length = np.sum(np.linalg.norm(np.diff(learned_traj, axis=0), axis=1))
    path_length_ratio = learned_path_length / (real_path_length + 1e-9)
    
    metrics = {
        'mean_distance': mean_distance,
        'max_distance': max_distance,
        'std_distance': std_distance,
        'final_error': final_error,
        'path_length_ratio': path_length_ratio,
        'real_path_length': real_path_length,
        'learned_path_length': learned_path_length
    }
    
    return metrics


def evaluate_all_test_trajectories(real_traj_arrays, learned_trajectories, qp=False,
                                   h_fn=None, grad_fn=None,
                                   h_surface_fn=None, grad_surface_fn=None):
    """Evaluate controller on all test trajectories.
    
    Args:
        controller: Trained controller model
        trajectories_test: List of (start, goal, real_traj) tuples
        
    Returns:
        all_metrics: List of metrics dicts for each trajectory
        summary: Dict with aggregated statistics
    """
    print('\n========== EVALUATING TEST TRAJECTORIES ==========')
    
    all_metrics = []
    
    # If GT surface functions are provided, compute per-trajectory mean distance to surface
    per_traj_surface_means = []

    for real_traj, learned_traj in zip(real_traj_arrays, learned_trajectories):

        # Calculate metrics
        metrics = evaluate_trajectory_accuracy(real_traj, learned_traj)
        all_metrics.append(metrics)

        # If GT surface functions provided, compute approximate distance to surface
        if h_surface_fn is not None and grad_surface_fn is not None:
            # ensure learned_traj is array
            lt = np.asarray(learned_traj)
            if lt.size == 0:
                per_traj_surface_means.append(float('nan'))
            else:
                hs = []
                for x in lt:
                    try:
                        hval = float(h_surface_fn(x))
                    except Exception:
                        hval = float('nan')
                    try:
                        g = np.asarray(grad_surface_fn(x)).reshape(-1)
                        gnorm = float(np.linalg.norm(g))
                    except Exception:
                        gnorm = 0.0
                    if np.isnan(hval):
                        continue
                    if gnorm > 1e-12:
                        approx_dist = abs(hval) / gnorm
                    else:
                        approx_dist = abs(hval)
                    hs.append(approx_dist)
                if len(hs) > 0:
                    per_traj_surface_means.append(float(np.mean(hs)))
                else:
                    per_traj_surface_means.append(float('nan'))
        
        print(f'  Mean distance:     {metrics["mean_distance"]:.6f}')
        print(f'  Max distance:      {metrics["max_distance"]:.6f}')
        print(f'  Final error:       {metrics["final_error"]:.6f}')
        print(f'  Path length ratio: {metrics["path_length_ratio"]:.4f}')
    
    # Aggregate statistics
    summary = {
        'avg_mean_distance': np.mean([m['mean_distance'] for m in all_metrics]),
        'avg_max_distance': np.mean([m['max_distance'] for m in all_metrics]),
        'avg_final_error': np.mean([m['final_error'] for m in all_metrics]),
        'avg_path_length_ratio': np.mean([m['path_length_ratio'] for m in all_metrics]),
        'worst_mean_distance': np.max([m['mean_distance'] for m in all_metrics]),
        'best_mean_distance': np.min([m['mean_distance'] for m in all_metrics])
    }

    # If we computed per-trajectory surface means, include aggregate in summary
    if len(per_traj_surface_means) > 0:
        try:
            vals = [v for v in per_traj_surface_means if not (isinstance(v, float) and np.isnan(v))]
            summary['mean_surface_distance'] = float(np.mean(vals)) if len(vals) > 0 else float('nan')
        except Exception:
            summary['mean_surface_distance'] = float('nan')
    
    print('\n========== SUMMARY ACROSS ALL TEST TRAJECTORIES ==========')
    print(f'Average mean distance:     {summary["avg_mean_distance"]:.6f}')
    print(f'Average max distance:      {summary["avg_max_distance"]:.6f}')
    print(f'Average final error:       {summary["avg_final_error"]:.6f}')
    print(f'Average path length ratio: {summary["avg_path_length_ratio"]:.4f}')
    print(f'Best trajectory distance:  {summary["best_mean_distance"]:.6f}')
    print(f'Worst trajectory distance: {summary["worst_mean_distance"]:.6f}')
    
    return summary
