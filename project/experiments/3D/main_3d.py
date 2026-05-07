"""
Main 3D experiment runner.
Consolidates training and visualization pipeline for all 3D constraints.

Usage:
    python main_3d.py --constraint sphere
    python main_3d.py --constraint spiral --archs NN
    python main_3d.py --constraint cylinder --noise 0.01 0.03 0.05
    python main_3d.py --constraint hyperplane --dataset trajectory --traj-num 5 --traj-length 4.0
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Setup paths
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.models3D import *
from src.training.train_loop import train_one_with_split
from src.data.datasets3D import *
from src.data.datasets3D import generate_surface_trajectory
from src.data.datasets3D import (
    sample_negative_from_estimated_normals,
    sample_negative_spiral_estimated_normals
)
from src.utils.viz import *
from src.utils.viz3D import *
from src.utils.metrics import compute_chamfer_distance
from src.utils.eval import eval_accuracy_simple
from configs import config3D as config
from src.utils.seed import set_seed

set_seed(config.SEED)


# ============================================================================
# Utility functions
# ============================================================================

def compute_projection_success_rate(model, test_points, threshold=0.01, max_iters=50, lr=0.1):
    """
    Compute projection success rate (comparable to ECoMaNN paper).
    
    Projects points to learned surface using gradient descent.
    Success = |f(x_proj)| < threshold after convergence.
    
    Args:
        model: Trained model
        test_points: Points to project (numpy array or tensor)
        threshold: Success threshold for |f(x)| (default: 0.01)
        max_iters: Maximum projection iterations (default: 50)
        lr: Learning rate for gradient descent (default: 0.1)
        
    Returns:
        success_rate: Percentage of successful projections [0-100]
    """
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(test_points, np.ndarray):
        points = torch.from_numpy(test_points).float().clone()
    else:
        points = test_points.clone()
    
    points.requires_grad = True
    
    successful = 0
    total = len(points)
    
    with torch.enable_grad():
        for iter_idx in range(max_iters):
            # Evaluate function
            f_vals = model(points)
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=f_vals.sum(),
                inputs=points,
                create_graph=False
            )[0]
            
            # Gradient descent step: x_new = x - lr * f(x) * grad(f) / |grad(f)|
            grad_norms = grads.norm(dim=1, keepdim=True).clamp(min=1e-6)
            points = points - lr * f_vals.unsqueeze(1) * grads / grad_norms
            points = points.detach().requires_grad_(True)
    
    # Final evaluation
    with torch.no_grad():
        final_f_vals = model(points).abs()
        successful = (final_f_vals < threshold).sum().item()
    
    success_rate = 100.0 * successful / total
    return success_rate


def sanitize_filename(name):
    """Sanitize a string for use in filenames by removing/replacing invalid characters."""
    import re
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Replace other potentially problematic characters
    name = name.replace('(', '').replace(')', '').replace('+', 'plus')
    # Remove any characters that are invalid in Windows filenames
    # Invalid: < > : " / \ | ? *
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Remove any non-ASCII characters
    name = name.encode('ascii', 'ignore').decode('ascii')
    # Convert to lowercase for consistency
    name = name.lower().strip()
    # Remove any trailing dots or spaces (Windows doesn't allow)
    name = name.rstrip('. ')
    return name


# ============================================================================
# Architecture definitions
# ============================================================================

ARCHITECTURES = {
    "NN": (lambda: SpiralImplicitMLP(), config.EPOCHS, config.LR_NN),
    "ParamQuad": (lambda: ParametricQuad(), config.EPOCHS, config.LR_POLY),
    "ParamLinear": (lambda: ParametricLinear(), config.EPOCHS, config.LR_POLY),
    "Hybrid": (lambda: HybridModel3D(SpiralImplicitMLP(), AnalyticalSpiral()), config.EPOCHS, config.LR_NN),
    "AnalyticalSpiral": (lambda: AnalyticalSpiral(), config.EPOCHS, config.LR_POLY),
}

# Loss variants (only keep the ones we want)
LOSS_VARIANTS = [
    #(2,"Margin loss"),
    #(5, "KNN Loss"),
    (6, "Pull Loss")
]


# ============================================================================
# Constraint-specific configurations
# ============================================================================

CONSTRAINT_CONFIGS = {
    "sphere": {
        "archs": ["NN"],  # Default architectures for sphere
        "bounds": [-1.3, 1.3, -1.3, 1.3, -1.3, 1.3],
    },
    "ellipsoid": {
        "archs": ["NN"],
        "bounds": [-1.6, 1.6, -1.6, 1.6, -1.6, 1.6],
    },
    "spiral": {
        "archs": ["NN"],
        "bounds": [-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    },
    "spiral_tube": {
        "archs": ["NN"],
        "bounds": [-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    },
    "cylinder": {
        "archs": ["NN"],
        "bounds": [-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    },
    "hyperplane": {
        "archs": ["NN"],
        "bounds": [-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    },
    "hyperboloid": {
        "archs": ["NN"],
        "bounds": [-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    },
}


# ============================================================================
# Main training and visualization pipeline
# ============================================================================

def run_experiment(constraint_name, archs=None, noise_levels=None, enable_plots=None, 
                   dataset_type="standard", traj_num=17, traj_length=4.0, traj_points=49,
                   show_plots=True, global_results=None, regularizers=None, cv_folds=1,
                   traj_holdout=0.0, eval_heldout=False):
    """
    Run complete 3D experiment for a given constraint.
    
    Args:
        constraint_name: Name of constraint (sphere, spiral, cylinder, hyperplane)
        archs: List of architecture names to use (default: from config)
        noise_levels: List of noise levels (default: [0.03])
        enable_plots: Dict of which plots to generate (default: all enabled)
        dataset_type: "standard" or "trajectory" - type of dataset to use
        traj_num: Number of trajectories to generate (for trajectory dataset)
        traj_length: Total length of each trajectory (for trajectory dataset)
        traj_points: Number of points per trajectory (for trajectory dataset)
        show_plots: If False, saves plots without displaying them (default: True)
        global_results: Dict to accumulate results across all constraints (optional)
        regularizers: List of regularizer types to test (['none', 'l2', 'eikonal'])
        cv_folds: Number of cross-validation folds. 1 disables CV (default: 1)
        traj_holdout: Fraction of trajectories to hold out from training (e.g., 0.2 for 80/20)
        eval_heldout: If True and trajectory dataset with holdout, also evaluate models on held-out trajectories
    
    Returns:
        dict: Summary of results for this constraint
    """
    # Get constraint-specific config
    if constraint_name not in CONSTRAINT_CONFIGS:
        raise ValueError(f"Unknown constraint: {constraint_name}. Available: {list(CONSTRAINT_CONFIGS.keys())}")
    
    cfg = CONSTRAINT_CONFIGS[constraint_name]
    
    # Use default architectures if not specified
    if archs is None:
        archs = cfg["archs"]
    
    # Default regularizers
    if regularizers is None:
        regularizers = ['none']  # Default: no regularization    # Convert arch names to tuples
    arch_configs = [(name, *ARCHITECTURES[name]) for name in archs]
    
    # Default noise levels
    if noise_levels is None:
        noise_levels = [0.03]
    
    # Default plots (all enabled)
    if enable_plots is None:
        enable_plots = {
            "curves": True,
            "heatmap": False,
            "confusion": False,
            "gt_samples": True,
            "contours": True,
            "projection": True,
            "slices": True,
        }
    
    print(f"\n{'='*80}")
    print(f"Running 3D Experiment: {constraint_name.upper()}")
    print(f"Dataset type: {dataset_type.upper()}")
    print(f"Architectures: {archs}")
    print(f"Noise levels: {noise_levels}")
    print(f"Regularizers: {regularizers}")
    print(f"CV folds: {cv_folds}")
    if dataset_type == "trajectory":
        print(f"Trajectory holdout: {traj_holdout}")
    print(f"Display plots: {'Yes' if show_plots else 'No (save only)'}")
    if dataset_type == "trajectory":
        print(f"Trajectory config: {traj_num} trajectories × {traj_points} points, length={traj_length}")
    print(f"{'='*80}\n")
    
    # Reset seed for reproducibility of this specific constraint
    # This ensures results are reproducible regardless of experiment order
    set_seed(config.SEED)
    
    # Set matplotlib backend based on show_plots
    if not show_plots:
        matplotlib.use('Agg')  # Non-interactive backend
    
    # Create ground truth functions
    h_np, grad_np, gt_samples, gt_curve = make_constraint_3d(constraint_name)
    gt_samples_3d = gt_samples(1000)
    # Generate samples based on dataset type
    if dataset_type == "trajectory":
        print(f"Generating {traj_num} trajectories on {constraint_name} surface...")
        trajectories = []
        
        for i in range(traj_num):
            print(f"  Trajectory {i+1}/{traj_num}...", end=" ")
            # pass a deterministic per-trajectory seed so generated trajectories
            # are reproducible across runs while still giving different trajectories
            traj_seed = int(config.SEED+1) + i
            traj_points_data = generate_surface_trajectory(
                total_length=traj_length,
                num_steps=traj_points,
                h=h_np,
                grad=grad_np,
                get_samples=gt_samples,
                seed=traj_seed
            )
            
            if len(traj_points_data) > 0:
                trajectories.append(traj_points_data)
                print(f"✓ {len(traj_points_data)} points")
            else:
                print("✗ Empty trajectory")
        
        if len(trajectories) == 0:
            raise ValueError("No valid trajectories could be generated!")
        
        # Trajectory-level holdout split (deterministic)
        rng = np.random.RandomState(int(config.SEED))
        idx_all = np.arange(len(trajectories))
        rng.shuffle(idx_all)
        n_hold = int(round(len(trajectories) * max(0.0, min(1.0, float(traj_holdout)))))
        hold_ids = idx_all[:n_hold]
        train_ids = idx_all[n_hold:]

        traj_holdout_list = [trajectories[i] for i in hold_ids]
        traj_train = [trajectories[i] for i in train_ids] if len(train_ids) > 0 else []

        if len(traj_train) == 0:
            raise ValueError("No training trajectories after holdout. Reduce --traj-holdout.")

        # Concatenate all training trajectories
        pos_np = np.concatenate(traj_train, axis=0)
        print(f"  Trajectories: train={len(traj_train)} holdout={len(traj_holdout_list)} (frac={traj_holdout})")
        print(f"  Total training trajectory points (before noise): {len(pos_np)}")
    else:
        # Standard sampling
        if constraint_name == "spiral_tube":
            pos_np = gt_samples(config.N_SPIRAL_TUBE_SAMPLES)
        else:
            pos_np = gt_samples(config.NPOS)
    
    # NOTE: Negatives will be generated AFTER noise is applied in the training loop
    # This ensures PCA local has access to all noisy points for better normal estimation
    neg_np = None  # Placeholder - will be generated later with noise applied
    
    # Generate test set based on dataset type:
    # - Standard dataset: test comes from same distribution (split later)
    # - Trajectory dataset: test is ALWAYS uniform (for fair comparison)
    if dataset_type == "trajectory":
        print(f"\nGenerating UNIFORM test set for trajectory evaluation...")
        pos_test_uniform = gt_samples(500)  # Fresh uniform samples for testing
        if constraint_name == "spiral":
            neg_test_uniform = sample_negative_spiral_estimated_normals(pos_test_uniform, std=0.4, k_neighbors=10)
        elif constraint_name == "spiral_tube":
            neg_test_uniform = sample_negative_spiral_estimated_normals(pos_test_uniform, std=0.4, k_neighbors=10)
        else:
            neg_test_uniform = sample_negative_from_estimated_normals(pos_test_uniform, std=0.30, k_neighbors=10)
        print(f"  ✓ Uniform test set: {len(pos_test_uniform)} pos, {len(neg_test_uniform)} neg")
        print(f"  NOTE: Training with trajectories, testing with uniform sampling")
    else:
        pos_test_uniform = None
        neg_test_uniform = None
    
    # Setup output directory
    output_dir = os.path.join(HERE, constraint_name, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    # If we generated trajectories for training, save them so they can be
    # re-used later (e.g. to train / evaluate a controller without re-sampling).
    if dataset_type == "trajectory":
        try:
            import pickle
            traj_train_save_name = f"training_trajectories_traj{traj_num}_pts{traj_points}.pkl"
            traj_save_path = os.path.join(output_dir, traj_train_save_name)
            with open(traj_save_path, 'wb') as f:
                pickle.dump(traj_train, f)
            print(f"  ✓ Saved training trajectories to: {traj_save_path}")
            if 'traj_holdout_list' in locals() and len(traj_holdout_list) > 0:
                traj_hold_save_name = f"holdout_trajectories_traj{len(traj_holdout_list)}_pts{traj_points}.pkl"
                traj_hold_save_path = os.path.join(output_dir, traj_hold_save_name)
                with open(traj_hold_save_path, 'wb') as f:
                    pickle.dump(traj_holdout_list, f)
                print(f"  ✓ Saved holdout trajectories to: {traj_hold_save_path}")
        except Exception as e:
            print(f"  Warning: could not save training trajectories: {e}")
    
    # Helper: K-fold indices generator
    def _make_kfold_indices(n, k, seed_val):
        idx = np.arange(n)
        rng = np.random.RandomState(int(seed_val))
        rng.shuffle(idx)
        return np.array_split(idx, int(k))

    # Run experiments for each noise level and architecture combination
    for noise in noise_levels:
        for arch_name, make_model, epochs, lr in arch_configs:
            print(f"\n{'─'*80}")
            print(f"Training with noise level: {noise}, architecture: {arch_name}")
            print(f"{'─'*80}")
            
            results = {}
            projection_infos = {}
            # For CV runs we also keep per-fold results to aggregate metrics
            per_fold_models = {}  # key: (arch, vname) -> [models per fold]
            per_fold_metrics = {}  # key: (arch, vname) -> list of metrics dicts

            # Create architecture-specific config for this single architecture
            single_arch_config = [(arch_name, make_model, epochs, lr)]

            # We'll collect the loss variants we actually ran (for plotting/CSV)
            executed_loss_variants = []

            # Train for each loss variant. For Neural Pull (6) we run each requested regularizer;
            # for other variants (Margin, KNN) we run the base training once.
            for vid, vname in LOSS_VARIANTS:
                if vid == 6:
                    # Neural Pull: run for each regularizer
                    for regularizer in regularizers:
                        # Reset seed for reproducibility between regularizers
                        set_seed(config.SEED)

                        reg_display_name = {
                            'none': 'Pull Loss',
                            'l2': 'Pull Loss + L2',
                            'eikonal': 'Pull Loss + Eikonal'
                        }.get(regularizer, regularizer)

                        print(f"  Training {arch_name} with {reg_display_name}...")

                        # Cross-validation logic
                        k = int(cv_folds) if cv_folds and cv_folds > 1 else 1
                        fold_models = []
                        fold_train_curves = []
                        fold_test_curves = []
                        fold_projection_infos = []

                        if k > 1:
                            # Trajectory-aware folds if dataset is trajectory; else point-wise folds
                            if dataset_type == "trajectory":
                                num_traj = len(traj_train)
                                traj_ids = np.arange(num_traj)
                                rng = np.random.RandomState(int(config.SEED))
                                rng.shuffle(traj_ids)
                                traj_folds = np.array_split(traj_ids, k)
                            else:
                                pos_folds = _make_kfold_indices(len(pos_np), k, config.SEED)
                                # Note: neg_folds not needed - negatives generated from positives after noise

                            for f in range(k):
                                set_seed(config.SEED + f)
                                model = make_model()

                                if dataset_type == "trajectory":
                                    # Build train set by concatenating all trajectories not in fold f
                                    train_traj_ids = np.concatenate([traj_folds[i] for i in range(k) if i != f])
                                    pos_list = [traj_train[t] for t in train_traj_ids]
                                    if len(pos_list) == 0:
                                        continue
                                    pos_train_np = np.concatenate(pos_list, axis=0)
                                    
                                    # IMPORTANT: Add noise FIRST
                                    if noise and noise > 0.0:
                                        pos_train_np_noisy = pos_train_np + np.random.normal(0.0, noise, size=pos_train_np.shape)
                                    else:
                                        pos_train_np_noisy = pos_train_np.copy()
                                    
                                    # THEN generate negatives using ALL noisy points (better PCA estimation)
                                    print(f"    Generating negatives from {len(pos_train_np_noisy)} noisy points...")
                                    if constraint_name == "spiral":
                                        neg_train_np = sample_negative_spiral_estimated_normals(pos_train_np_noisy, std=0.4, k_neighbors=10)
                                    elif constraint_name == "spiral_tube":
                                        neg_train_np = sample_negative_spiral_estimated_normals(pos_train_np_noisy, std=0.4, k_neighbors=10)
                                    else:
                                        neg_train_np = sample_negative_from_estimated_normals(pos_train_np_noisy, std=0.30, k_neighbors=10)
                                    
                                    pos_train = torch.from_numpy(pos_train_np_noisy).float()
                                    neg_train = torch.from_numpy(neg_train_np).float()
                                else:
                                    # Standard dataset: point-wise folds
                                    # Build train indices: all except fold f
                                    pos_train_idx = np.concatenate([pos_folds[i] for i in range(k) if i != f]) if k > 1 else np.arange(len(pos_np))
                                    pos_train_np = pos_np[pos_train_idx]
                                    
                                    # Add noise FIRST
                                    if noise and noise > 0.0:
                                        pos_train_np_noisy = pos_train_np + np.random.normal(0.0, noise, size=pos_train_np.shape)
                                    else:
                                        pos_train_np_noisy = pos_train_np.copy()
                                    
                                    # Generate negatives from noisy positives
                                    print(f"    Generating negatives from {len(pos_train_np_noisy)} noisy points...")
                                    if constraint_name == "spiral":
                                        neg_train_np = sample_negative_spiral_estimated_normals(pos_train_np_noisy, std=0.4, k_neighbors=10)
                                    elif constraint_name == "spiral_tube":
                                        neg_train_np = sample_negative_spiral_estimated_normals(pos_train_np_noisy, std=0.4, k_neighbors=10)
                                    else:
                                        neg_train_np = sample_negative_from_estimated_normals(pos_train_np_noisy, std=0.30, k_neighbors=10)
                                    
                                    pos_train = torch.from_numpy(pos_train_np_noisy).float()
                                    neg_train = torch.from_numpy(neg_train_np).float()

                                # Eval set for training loop
                                if dataset_type == "trajectory":
                                    pos_eval = torch.from_numpy(pos_test_uniform).float()
                                    neg_eval = torch.from_numpy(neg_test_uniform).float()
                                else:
                                    # Use fold-out as validation for standard dataset
                                    pos_eval_idx = pos_folds[f]
                                    # If a fold is empty, fallback to uniform split like before
                                    if len(pos_eval_idx) == 0:
                                        pt, pe, nt, ne = split_train_test_3d(pos_np, None, eval_split=config.EVAL_SPLIT, noise_level=0.0)
                                        pos_eval_np = pe.cpu().numpy()
                                    else:
                                        pos_eval_np = pos_np[pos_eval_idx]
                                    
                                    # Generate negatives for eval set WITHOUT noise (clean eval)
                                    if constraint_name == "spiral":
                                        neg_eval_np = sample_negative_spiral_estimated_normals(pos_eval_np, std=0.4, k_neighbors=10)
                                    elif constraint_name == "spiral_tube":
                                        neg_eval_np = sample_negative_spiral_estimated_normals(pos_eval_np, std=0.4, k_neighbors=10)
                                    else:
                                        neg_eval_np = sample_negative_from_estimated_normals(pos_eval_np, std=0.30, k_neighbors=10)
                                    
                                    pos_eval = torch.from_numpy(pos_eval_np).float()
                                    neg_eval = torch.from_numpy(neg_eval_np).float()

                                model, train_curve, test_curve, projection_info = train_one_with_split(
                                    model, pos_train, neg_train, pos_eval, neg_eval,
                                    variant=6, epochs=epochs, lr=lr,
                                    batch=config.BATCH, margin=config.MARGIN, alpha=config.ALPHA_NEG,
                                    lam_l2=config.LAM_L2, lam_eikonal=config.LAM_EIKONAL,
                                    regularizer=regularizer
                                )

                                fold_models.append(model)
                                fold_train_curves.append(train_curve)
                                fold_test_curves.append(test_curve)
                                fold_projection_infos.append(projection_info)
                                if f == 0:
                                    projection_infos[f"{arch_name}_{regularizer}"] = projection_info

                            # Use fold 1 artifacts for plotting/curves; keep models for evaluation
                            model_plot = fold_models[0]
                            train_curve_plot = fold_train_curves[0]
                            test_curve_plot = fold_test_curves[0]
                            results[(arch_name, reg_display_name)] = (model_plot, train_curve_plot, test_curve_plot)
                            per_fold_models[(arch_name, reg_display_name)] = fold_models
                            executed_loss_variants.append((6, reg_display_name))
                        else:
                            # Single run (no CV)
                            model = make_model()

                            if dataset_type == "trajectory":
                                # Add noise to positives FIRST
                                pos_train_np = pos_np.copy()
                                if noise and noise > 0.0:
                                    pos_train_np = pos_train_np + np.random.normal(0.0, noise, size=pos_train_np.shape)
                                
                                # Generate negatives from ALL noisy trajectory points
                                print(f"    Generating negatives from {len(pos_train_np)} noisy trajectory points...")
                                if constraint_name == "spiral":
                                    neg_train_np = sample_negative_spiral_estimated_normals(pos_train_np, std=0.4, k_neighbors=10)
                                elif constraint_name == "spiral_tube":
                                    neg_train_np = sample_negative_spiral_estimated_normals(pos_train_np, std=0.4, k_neighbors=10)
                                else:
                                    neg_train_np = sample_negative_from_estimated_normals(pos_train_np, std=0.30, k_neighbors=10)
                                
                                pos_train = torch.from_numpy(pos_train_np).float()
                                neg_train = torch.from_numpy(neg_train_np).float()
                                pos_eval = torch.from_numpy(pos_test_uniform).float()
                                neg_eval = torch.from_numpy(neg_test_uniform).float()
                            else:
                                # Standard dataset: add noise first, then generate negatives
                                pos_np_noisy = pos_np.copy()
                                if noise and noise > 0.0:
                                    pos_np_noisy = pos_np_noisy + np.random.normal(0.0, noise, size=pos_np_noisy.shape)
                                
                                # Generate negatives from ALL noisy points
                                print(f"    Generating negatives from {len(pos_np_noisy)} noisy points...")
                                if constraint_name == "spiral":
                                    neg_np = sample_negative_spiral_estimated_normals(pos_np_noisy, std=0.4, k_neighbors=10)
                                elif constraint_name == "spiral_tube":
                                    neg_np = sample_negative_spiral_estimated_normals(pos_np_noisy, std=0.4, k_neighbors=10)
                                else:
                                    neg_np = sample_negative_from_estimated_normals(pos_np_noisy, std=0.30, k_neighbors=10)
                                
                                # Now split (without adding more noise)
                                pos_train, pos_eval, neg_train, neg_eval = split_train_test_3d(
                                    pos_np_noisy, neg_np, eval_split=config.EVAL_SPLIT, noise_level=0.0
                                )

                            model, train_curve, test_curve, projection_info = train_one_with_split(
                                model, pos_train, neg_train, pos_eval, neg_eval,
                                variant=6, epochs=epochs, lr=lr,
                                batch=config.BATCH, margin=config.MARGIN, alpha=config.ALPHA_NEG,
                                lam_l2=config.LAM_L2, lam_eikonal=config.LAM_EIKONAL,
                                regularizer=regularizer
                            )
                            projection_infos[f"{arch_name}_{regularizer}"] = projection_info
                            results[(arch_name, reg_display_name)] = (model, train_curve, test_curve)
                            executed_loss_variants.append((6, reg_display_name))
                        # Report last/avg loss quickly
                        if k > 1:
                            last_train = fold_train_curves[-1][-1] if fold_train_curves and fold_train_curves[-1] else 0
                            last_test = fold_test_curves[-1][-1] if fold_test_curves and fold_test_curves[-1] else 0
                            print(f"    CV last fold train loss: {last_train:.6f}, test loss: {last_test:.6f}")
                        else:
                            print(f"    Final train loss: {train_curve[-1] if train_curve else 0:.6f}, test loss: {test_curve[-1] if test_curve else 0:.6f}")
                else:
                    # Other losses (e.g. Margin, KNN): run once each
                    set_seed(config.SEED)
                    print(f"  Training {arch_name} with {vname} (variant {vid})...")
                    # Cross-validation logic for non-projection variants
                    k = int(cv_folds) if cv_folds and cv_folds > 1 else 1
                    if k > 1:
                        # Trajectory-aware folds if trajectory dataset
                        if dataset_type == "trajectory":
                            num_traj = len(traj_train)
                            traj_ids = np.arange(num_traj)
                            rng = np.random.RandomState(int(config.SEED))
                            rng.shuffle(traj_ids)
                            traj_folds = np.array_split(traj_ids, k)
                        else:
                            pos_folds = _make_kfold_indices(len(pos_np), k, config.SEED)
                            # Note: neg_folds not needed - negatives generated from positives after noise

                        fold_models = []
                        fold_train_curves = []
                        fold_test_curves = []
                        for f in range(k):
                            set_seed(config.SEED + f)
                            model = make_model()

                            if dataset_type == "trajectory":
                                train_traj_ids = np.concatenate([traj_folds[i] for i in range(k) if i != f])
                                pos_list = [traj_train[t] for t in train_traj_ids]
                                if len(pos_list) == 0:
                                    continue
                                pos_train_np = np.concatenate(pos_list, axis=0)
                                # Add noise to positives first
                                if config.NOISE_LEVEL and config.NOISE_LEVEL > 0.0:
                                    pos_train_noisy = pos_train_np + np.random.normal(0.0, config.NOISE_LEVEL, size=pos_train_np.shape)
                                else:
                                    pos_train_noisy = pos_train_np.copy()
                                # Generate negatives from all noisy positives (better PCA with all points)
                                if constraint_name == "spiral":
                                    neg_train_np = sample_negative_spiral_estimated_normals(pos_train_noisy, std=0.4, k_neighbors=10)
                                elif constraint_name == "spiral_tube":
                                    neg_train_np = sample_negative_spiral_estimated_normals(pos_train_noisy, std=0.4, k_neighbors=10)
                                else:
                                    neg_train_np = sample_negative_from_estimated_normals(pos_train_noisy, std=0.30, k_neighbors=10)
                                pos_train = torch.from_numpy(pos_train_noisy).float()
                                neg_train = torch.from_numpy(neg_train_np).float()
                            else:
                                # Standard dataset: add noise first, then generate negatives
                                # Get train fold indices
                                pos_train_idx = np.concatenate([pos_folds[i] for i in range(k) if i != f]) if k > 1 else np.arange(len(pos_np))
                                pos_train_np = pos_np[pos_train_idx]
                                
                                if config.NOISE_LEVEL and config.NOISE_LEVEL > 0.0:
                                    pos_train_noisy = pos_train_np + np.random.normal(0.0, config.NOISE_LEVEL, size=pos_train_np.shape)
                                else:
                                    pos_train_noisy = pos_train_np.copy()
                                
                                # Generate negatives from noisy train positives
                                if constraint_name == "spiral":
                                    neg_train_np = sample_negative_spiral_estimated_normals(pos_train_noisy, std=0.4, k_neighbors=10)
                                elif constraint_name == "spiral_tube":
                                    neg_train_np = sample_negative_spiral_estimated_normals(pos_train_noisy, std=0.4, k_neighbors=10)
                                else:
                                    neg_train_np = sample_negative_from_estimated_normals(pos_train_noisy, std=0.30, k_neighbors=10)
                                
                                pos_train = torch.from_numpy(pos_train_noisy).float()
                                neg_train = torch.from_numpy(neg_train_np).float()

                            # Eval set for training loop
                            if dataset_type == "trajectory":
                                pos_eval = torch.from_numpy(pos_test_uniform).float()
                                neg_eval = torch.from_numpy(neg_test_uniform).float()
                            else:
                                # Use fold-out as validation for standard dataset
                                pos_eval_idx = pos_folds[f]
                                # If a fold is empty, fallback to uniform split
                                if len(pos_eval_idx) == 0:
                                    pt, pe, nt, ne = split_train_test_3d(pos_np, None, eval_split=config.EVAL_SPLIT, noise_level=0.0)
                                    pos_eval_np = pe.cpu().numpy()
                                else:
                                    pos_eval_np = pos_np[pos_eval_idx]
                                
                                # Generate negatives for eval set WITHOUT noise (clean eval)
                                if constraint_name == "spiral":
                                    neg_eval_np = sample_negative_spiral_estimated_normals(pos_eval_np, std=0.4, k_neighbors=10)
                                elif constraint_name == "spiral_tube":
                                    neg_eval_np = sample_negative_spiral_estimated_normals(pos_eval_np, std=0.4, k_neighbors=10)
                                else:
                                    neg_eval_np = sample_negative_from_estimated_normals(pos_eval_np, std=0.30, k_neighbors=10)
                                
                                pos_eval = torch.from_numpy(pos_eval_np).float()
                                neg_eval = torch.from_numpy(neg_eval_np).float()

                            model, train_curve, test_curve = train_one_with_split(
                                model, pos_train, neg_train, pos_eval, neg_eval,
                                variant=vid, epochs=epochs, lr=lr,
                                batch=config.BATCH, margin=config.MARGIN, alpha=config.ALPHA_NEG
                            )
                            fold_models.append(model)
                            fold_train_curves.append(train_curve)
                            fold_test_curves.append(test_curve)

                        # Use fold 1 artifacts for plotting/curves; keep models for evaluation
                        results[(arch_name, vname)] = (fold_models[0], fold_train_curves[0], fold_test_curves[0])
                        per_fold_models[(arch_name, vname)] = fold_models
                        executed_loss_variants.append((vid, vname))
                        last_train = fold_train_curves[-1][-1] if fold_train_curves and fold_train_curves[-1] else 0
                        last_test = fold_test_curves[-1][-1] if fold_test_curves and fold_test_curves[-1] else 0
                        print(f"    CV last fold train loss: {last_train:.6f}, test loss: {last_test:.6f}")
                    else:
                        model = make_model()
                        if dataset_type == "trajectory":
                            # Add noise to positives first
                            if noise and noise > 0.0:
                                pos_np_noisy = pos_np + np.random.normal(0.0, noise, size=pos_np.shape)
                            else:
                                pos_np_noisy = pos_np.copy()
                            # Generate negatives from all noisy positives
                            if constraint_name == "spiral":
                                neg_np_full = sample_negative_spiral_estimated_normals(pos_np_noisy, std=0.4, k_neighbors=10)
                            elif constraint_name == "spiral_tube":
                                neg_np_full = sample_negative_spiral_estimated_normals(pos_np_noisy, std=0.4, k_neighbors=10)
                            else:
                                neg_np_full = sample_negative_from_estimated_normals(pos_np_noisy, std=0.30, k_neighbors=10)
                            
                            # For trajectory: use all training data (no eval split)
                            pos_train = torch.from_numpy(pos_np_noisy).float()
                            neg_train = torch.from_numpy(neg_np_full).float()
                            pos_eval = torch.from_numpy(pos_test_uniform).float()
                            neg_eval = torch.from_numpy(neg_test_uniform).float()
                        else:
                            # Standard dataset: add noise first, then generate negatives, then split
                            if noise and noise > 0.0:
                                pos_np_noisy = pos_np + np.random.normal(0.0, noise, size=pos_np.shape)
                            else:
                                pos_np_noisy = pos_np.copy()
                            if constraint_name == "spiral":
                                neg_np_full = sample_negative_spiral_estimated_normals(pos_np_noisy, std=0.4, k_neighbors=10)
                            elif constraint_name == "spiral_tube":
                                neg_np_full = sample_negative_spiral_estimated_normals(pos_np_noisy, std=0.4, k_neighbors=10)
                            else:
                                neg_np_full = sample_negative_from_estimated_normals(pos_np_noisy, std=0.30, k_neighbors=10)
                            
                            # Split with noise_level=0.0 since noise already applied
                            pos_train, pos_eval, neg_train, neg_eval = split_train_test_3d(
                                pos_np_noisy, neg_np_full, eval_split=config.EVAL_SPLIT, noise_level=0.0
                            )
                        model, train_curve, test_curve = train_one_with_split(
                            model, pos_train, neg_train, pos_eval, neg_eval,
                            variant=vid, epochs=epochs, lr=lr,
                            batch=config.BATCH, margin=config.MARGIN, alpha=config.ALPHA_NEG
                        )
                        print(f"    Final train loss: {train_curve[-1] if train_curve else 0:.6f}, test loss: {test_curve[-1] if test_curve else 0:.6f}")
                        results[(arch_name, vname)] = (model, train_curve, test_curve)
                        executed_loss_variants.append((vid, vname))
            
            # This is independent of both training and test sets used during training
            print(f"\n  Generating FRESH uniform evaluation set for final metrics and plots...")
            pos_final_eval = gt_samples(500)
            if constraint_name == "spiral":
                neg_final_eval = sample_negative_spiral_estimated_normals(pos_final_eval, std=0.4, k_neighbors=10)
            elif constraint_name == "spiral_tube":
                neg_final_eval = sample_negative_spiral_estimated_normals(pos_final_eval, std=0.4, k_neighbors=10)
            else:
                neg_final_eval = sample_negative_from_estimated_normals(pos_final_eval, std=0.30, k_neighbors=10)
            print(f"  ✓ Fresh evaluation set: {len(pos_final_eval)} pos, {len(neg_final_eval)} neg")
            
            # Convert to tensors for metrics (uniform final eval set)
            pos_final_tensor = torch.from_numpy(pos_final_eval).float()
            neg_final_tensor = torch.from_numpy(neg_final_eval).float()
            
            # Compute metrics
            print(f"\n  Computing evaluation metrics...")
            print(f"    Chamfer Distance: Surface geometry (lower = better)")
            print(f"    Projection Success Rate: Convergence robustness (higher = better, comparable to ECoMaNN)")
            if dataset_type == "trajectory":
                print(f"    NOTE: Training with trajectories, evaluation with uniform sampling")
            
            metrics_results = {}
            
            # Prepare CSV for per-fold metrics if CV is enabled
            metrics_folds_rows = []

            for (arch_name_key, vname), (model, train_curve, test_curve) in results.items():
                print(f"\n    Evaluating {arch_name_key} - {vname}...")
                
                k = int(cv_folds) if cv_folds and cv_folds > 1 else 1
                if k > 1 and (arch_name_key, vname) in per_fold_models:
                    ch_list = []
                    acc_list = []
                    pos_acc_list = []
                    neg_acc_list = []
                    proj_list = []
                    # Evaluate each fold model on uniform set
                    for fold_idx, m in enumerate(per_fold_models[(arch_name_key, vname)]):
                        chamfer_f = compute_chamfer_distance(
                            m, gt_samples, level=0.0, bounds=cfg["bounds"], 
                            resolution=50, n_samples=5000
                        )
                        acc_d = eval_accuracy_simple(m, pos_final_tensor, neg_final_tensor, 
                                                     pos_eps=0.03, neg_thr=0.08)
                        proj_success = compute_projection_success_rate(m, pos_final_eval, threshold=0.01, max_iters=50)
                        ch_list.append(chamfer_f)
                        acc_list.append(acc_d['global_accuracy'])
                        pos_acc_list.append(acc_d['pos_accuracy'])
                        neg_acc_list.append(acc_d['neg_accuracy'])
                        proj_list.append(proj_success)
                        metrics_folds_rows.append([arch_name_key, vname, fold_idx+1,
                                                   f"{chamfer_f:.6f}",
                                                   f"{acc_d['global_accuracy']*100:.2f}",
                                                   f"{acc_d['pos_accuracy']*100:.2f}",
                                                   f"{acc_d['neg_accuracy']*100:.2f}",
                                                   f"{proj_success:.2f}"])

                    chamfer = float(np.mean(ch_list))
                    acc_global = float(np.mean(acc_list))
                    acc_pos = float(np.mean(pos_acc_list))
                    acc_neg = float(np.mean(neg_acc_list))
                    proj_success_rate = float(np.mean(proj_list))
                    print(f"      🎯 Chamfer (avg {k} folds) = {chamfer:.6f}")
                    print(f"      ✓ Accuracy (avg) = {acc_global*100:.2f}%")
                    print(f"      ✓ Projection Success Rate (avg) = {proj_success_rate:.2f}%")
                else:
                    # Single model evaluation (no CV)
                    chamfer = compute_chamfer_distance(
                        model, gt_samples, level=0.0, bounds=cfg["bounds"], 
                        resolution=50, n_samples=5000
                    )
                    acc_dict = eval_accuracy_simple(model, pos_final_tensor, neg_final_tensor, 
                                                   pos_eps=0.03, neg_thr=0.08)
                    proj_success_rate = compute_projection_success_rate(model, pos_final_eval, threshold=0.01, max_iters=50)
                    acc_global = acc_dict['global_accuracy']
                    acc_pos = acc_dict['pos_accuracy']
                    acc_neg = acc_dict['neg_accuracy']
                    print(f"      🎯 Chamfer = {chamfer:.6f}  (surface geometry)")
                    print(f"      ✓ Accuracy = {acc_global*100:.2f}% (classification)")
                    print(f"      ✓ Projection Success Rate = {proj_success_rate:.2f}% (comparable to ECoMaNN)")

                metrics_results[(arch_name_key, vname)] = {
                    'chamfer': chamfer,
                    'projection_success': proj_success_rate,
                    'total_acc': acc_global,
                    'pos_acc': acc_pos,
                    'neg_acc': acc_neg,
                    'train_loss': train_curve[-1] if train_curve else 0,
                    'test_loss': test_curve[-1] if test_curve else 0
                }

                # Optional: evaluation on held-out trajectories (classification metrics)
                if eval_heldout and dataset_type == "trajectory" and 'traj_holdout_list' in locals() and len(traj_holdout_list) > 0:
                    pos_holdout_np = np.concatenate(traj_holdout_list, axis=0)
                    # Generate negatives from held-out positives (no training leakage)
                    if constraint_name == "spiral":
                        neg_holdout_np = sample_negative_spiral_estimated_normals(pos_holdout_np, std=0.4, k_neighbors=10)
                    elif constraint_name == "spiral_tube":
                        neg_holdout_np = sample_negative_spiral_estimated_normals(pos_holdout_np, std=0.4, k_neighbors=10)
                    else:
                        neg_holdout_np = sample_negative_from_estimated_normals(pos_holdout_np, std=0.30, k_neighbors=10)

                    pos_holdout_t = torch.from_numpy(pos_holdout_np).float()
                    neg_holdout_t = torch.from_numpy(neg_holdout_np).float()

                    if int(cv_folds) > 1 and (arch_name_key, vname) in per_fold_models:
                        accs = []
                        accs_pos = []
                        accs_neg = []
                        for m in per_fold_models[(arch_name_key, vname)]:
                            acc_d_h = eval_accuracy_simple(m, pos_holdout_t, neg_holdout_t, pos_eps=0.03, neg_thr=0.08)
                            accs.append(acc_d_h['global_accuracy'])
                            accs_pos.append(acc_d_h['pos_accuracy'])
                            accs_neg.append(acc_d_h['neg_accuracy'])
                        held_acc = float(np.mean(accs))
                        held_pos = float(np.mean(accs_pos))
                        held_neg = float(np.mean(accs_neg))
                    else:
                        acc_d_h = eval_accuracy_simple(model, pos_holdout_t, neg_holdout_t, pos_eps=0.03, neg_thr=0.08)
                        held_acc = float(acc_d_h['global_accuracy'])
                        held_pos = float(acc_d_h['pos_accuracy'])
                        held_neg = float(acc_d_h['neg_accuracy'])

                    # Stash for CSV later
                    if 'heldout_metrics' not in locals():
                        heldout_metrics = {}
                    heldout_metrics[(arch_name_key, vname)] = {
                        'held_acc': held_acc,
                        'held_pos': held_pos,
                        'held_neg': held_neg,
                        'n_pos': int(pos_holdout_np.shape[0]),
                        'n_neg': int(neg_holdout_np.shape[0])
                    }
                
                # Store in global results if provided
                if global_results is not None:
                    # Extract loss type and regularizer from vname
                    loss_name = vname
                    reg_type = 'none'
                    
                    # Only extract regularizer if it's Neural Pull
                    if 'Neural Pull' in vname:
                        loss_name = 'Neural Pull'
                        if 'L2' in vname:
                            reg_type = 'l2'
                        elif 'Eikonal' in vname:
                            reg_type = 'eikonal'
                    
                    global_results.append({
                        'constraint': constraint_name,
                        'architecture': arch_name_key,
                        'loss': loss_name,
                        'regularizer': reg_type,
                        'noise': noise,
                        'dataset': dataset_type,
                        'chamfer': chamfer,
                        'projection_success': metrics_results[(arch_name_key, vname)]['projection_success'],
                        'accuracy': metrics_results[(arch_name_key, vname)]['total_acc'],
                        'pos_acc': metrics_results[(arch_name_key, vname)]['pos_acc'],
                        'neg_acc': metrics_results[(arch_name_key, vname)]['neg_acc'],
                        'train_loss': metrics_results[(arch_name_key, vname)]['train_loss'],
                        'test_loss': metrics_results[(arch_name_key, vname)]['test_loss']
                    })
            
            # Generate visualizations for this noise+architecture combination
            print(f"\n{'─'*80}")
            print(f"Generating plots for noise={noise}, arch={arch_name}, dataset={dataset_type}")
            print(f"{'─'*80}")
            
            # Create directory with different structure for regularizer experiments
            # If testing multiple regularizers or non-default, create special folder
            if len(regularizers) > 1 or (len(regularizers) == 1 and regularizers[0] != 'none'):
                # Regularizer experiments go in separate folder
                reg_folder_name = "_".join(regularizers)
                base_output = os.path.join(output_dir, "regularizer_experiments", reg_folder_name)
                
                if dataset_type == "trajectory":
                    fig_dir = os.path.join(base_output, f"noise_{noise}_{arch_name}_trajectory_{traj_num}")
                else:
                    fig_dir = os.path.join(base_output, f"noise_{noise}_{arch_name}")
            else:
                # Standard experiments (no regularizer or baseline)
                if dataset_type == "trajectory":
                    fig_dir = os.path.join(output_dir, f"noise_{noise}_{arch_name}_trajectory_{traj_num}")
                else:
                    fig_dir = os.path.join(output_dir, f"noise_{noise}_{arch_name}")

            # Append CV info to directory if applicable
            if cv_folds and cv_folds > 1:
                fig_dir = os.path.join(fig_dir, f"cv_{cv_folds}folds")
            
            os.makedirs(fig_dir, exist_ok=True)
            
            # Save trained models
            print(f"\n  Saving trained models...")
            models_dir = os.path.join(fig_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            for (arch_name_key, vname), (model, _, _) in results.items():
                safe_vname = sanitize_filename(vname)
                if cv_folds and cv_folds > 1 and (arch_name_key, vname) in per_fold_models:
                    # Save per-fold models
                    for fold_idx, m in enumerate(per_fold_models[(arch_name_key, vname)], start=1):
                        model_filename = f"{constraint_name}_{arch_name_key}_{safe_vname}_noise_{noise}_fold_{fold_idx}.pth"
                        model_path = os.path.join(models_dir, model_filename)
                        torch.save(m.state_dict(), model_path)
                        print(f"    ✓ {model_filename}")
                else:
                    # Single model
                    model_filename = f"{constraint_name}_{arch_name_key}_{safe_vname}_noise_{noise}.pth"
                    model_path = os.path.join(models_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f"    ✓ {model_filename}")
            
            print(f"  ✅ Models saved to: {models_dir}")
            
            # Save metrics to CSV
            import csv
            csv_path = os.path.join(fig_dir, "metrics_summary.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Architecture', 'Loss', 'Chamfer', 'Proj_Success(%)', 'Accuracy(%)', 
                               'Pos_Acc(%)', 'Neg_Acc(%)', 'Train_Loss', 'Test_Loss'])
                for (arch_name_key, vname), metrics in metrics_results.items():
                    writer.writerow([
                        arch_name_key, vname,
                        f"{metrics['chamfer']:.6f}",
                        f"{metrics['projection_success']:.2f}",
                        f"{metrics['total_acc']*100:.2f}",
                        f"{metrics['pos_acc']*100:.2f}",
                        f"{metrics['neg_acc']*100:.2f}",
                        f"{metrics['train_loss']:.6f}",
                        f"{metrics['test_loss']:.6f}"
                    ])
            print(f"  ✓ Metrics saved to: {csv_path}")
            # Save held-out trajectory evaluation metrics if available
            if eval_heldout and dataset_type == "trajectory" and 'heldout_metrics' in locals() and len(heldout_metrics) > 0:
                held_csv = os.path.join(fig_dir, "metrics_heldout_trajectories.csv")
                with open(held_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Architecture', 'Loss', 'Acc(%)', 'Pos_Acc(%)', 'Neg_Acc(%)', 'N_pos', 'N_neg'])
                    for (arch_name_key, vname), m in heldout_metrics.items():
                        writer.writerow([
                            arch_name_key, vname,
                            f"{m['held_acc']*100:.2f}",
                            f"{m['held_pos']*100:.2f}",
                            f"{m['held_neg']*100:.2f}",
                            m['n_pos'], m['n_neg']
                        ])
                print(f"  ✓ Held-out trajectory metrics saved to: {held_csv}")
            # If CV, also save per-fold metrics
            if cv_folds and cv_folds > 1 and metrics_folds_rows:
                folds_csv = os.path.join(fig_dir, "metrics_per_fold.csv")
                with open(folds_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Architecture', 'Loss', 'Fold', 'Chamfer', 'Accuracy(%)', 'Pos_Acc(%)', 'Neg_Acc(%)', 'Proj_Success(%)'])
                    writer.writerows(metrics_folds_rows)
                print(f"  ✓ Per-fold metrics saved to: {folds_csv}")
            print(f"  📊 PRIMARY: Chamfer (surface) | Proj Success (ECoMaNN comparison) | SECONDARY: Accuracy (classification)")
            
            # Print ranking
            sorted_results = sorted(metrics_results.items(), key=lambda x: x[1]['chamfer'])
            print(f"\n  🏆 RANKING BY CHAMFER:")
            for i, ((arch, loss), m) in enumerate(sorted_results, 1):
                print(f"     {i}. {loss:20s} Chamfer={m['chamfer']:.6f}  ProjSuccess={m['projection_success']:.2f}%")
            
            gt_samples_3d = gt_curve(M=1000)
            
            # Use the actual loss variants we executed for plotting and CSVs
            loss_variants_for_plots = executed_loss_variants
            
            # 1. Training curves
            if enable_plots["curves"]:
                print("  ✓ Training curves...")
                plot_curve_grid(
                    results, single_arch_config, loss_variants_for_plots,
                    save_path=os.path.join(fig_dir, "figB_curves.png")
                )
            
            # 2. Heatmap (using fresh evaluation set)
            if enable_plots["heatmap"]:
                print("  ✓ Heatmap...")
                plot_heatmap(
                    archs=single_arch_config,
                    loss_variants=loss_variants_for_plots,
                    name=constraint_name,
                    results=results,
                    pos_tensor=pos_final_tensor,
                    neg_tensor=neg_final_tensor,
                    config=config,
                    is_3d=True,
                    save_path=os.path.join(fig_dir, "figD_heatmap.png"),
                )
            
            # 3. Confusion matrix (using fresh evaluation set)
            if enable_plots["confusion"]:
                print("  ✓ Confusion matrix...")
                plot_confusion_matrix_grid(
                    results, single_arch_config, loss_variants_for_plots,
                    pos_final_tensor, neg_final_tensor,
                    save_path=os.path.join(fig_dir, "figE_confusion_matrices.png"),
                    pos_eps=0.03, neg_thr=0.08
                )
            
            # 4. GT with samples
            if enable_plots["gt_samples"]:
                print("  ✓ GT with samples...")
                plot_GT_with_samples_grid(
                    gt_samples=gt_samples_3d,
                    pos_np=pos_train,
                    neg_np=neg_train,
                    bounds=cfg["bounds"],
                    constraint_name=constraint_name,
                    gt_function=h_np,
                    save_path=os.path.join(fig_dir, "figI_GT_samples.png")
                )
            
            # 5. Surface contours (per regularizer)
            if enable_plots["contours"]:
                print("  ✓ Surface contours...")
                for vid, vname in loss_variants_for_plots:
                    results_subset = {}
                    if (arch_name, vname) in results:
                        results_subset[(arch_name, vname)] = results[(arch_name, vname)]
                    
                    if results_subset:
                        safe_vname = sanitize_filename(vname)
                        plot_GT_LS_Countours(
                            results=results_subset,
                            constraint_name=constraint_name,
                            bounds=cfg["bounds"],
                            gt_function=h_np,
                            save_path=os.path.join(fig_dir, f"figI_contours_{safe_vname}.png")
                        )
            
            # 6. Projection visualizations (one per regularizer)
            if enable_plots["projection"] and projection_infos:
                print("  ✓ Projection visualizations...")
                for reg in regularizers:
                    proj_key = f"{arch_name}_{reg}"
                    if proj_key in projection_infos:
                        reg_display_name = {
                            'none': 'no_reg',
                            'l2': 'l2',
                            'eikonal': 'eikonal'
                        }.get(reg, reg)
                        plot_projection_visualization_3d(
                            projection_infos[proj_key],
                            gt_samples_3d,
                            bounds=cfg["bounds"],
                            save_path=os.path.join(fig_dir, f"figJ_projection_{arch_name}_{reg_display_name}.png"),
                            n_show=50,
                            gt_function=h_np,
                            constraint_name=constraint_name
                        )
            
            # 7. 2D slices (automatic intelligent view selection)
            if enable_plots["slices"]:
                print("  ✓ 2D slice visualizations...")
                plot_3d_slices_grid(
                    results=results,
                    archs=single_arch_config,
                    loss_variants=loss_variants_for_plots,
                    gt_function=h_np,
                    bounds=cfg["bounds"],
                    resolution=160,
                    save_dir=os.path.join(fig_dir, "slices"),
                    constraint_name=constraint_name
                )
            
            print(f"\n  ✅ All plots saved to: {fig_dir}")
    
    print(f"\n{'='*80}")
    print(f"Experiment complete for {constraint_name.upper()}!")
    print(f"{'='*80}\n")


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run 3D implicit surface learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard dataset
  python main_3d.py --constraint sphere
  python main_3d.py --constraint spiral --archs NN
  python main_3d.py --constraint cylinder --noise 0.01 0.03 0.05
  
  # Trajectory dataset
  python main_3d.py --constraint hyperplane --dataset trajectory
  python main_3d.py --constraint sphere --dataset trajectory --traj-num 10 --traj-length 5.0
  python main_3d.py --constraint hyperplane --dataset trajectory --disable-plots projection slices
        """
    )
    
    parser.add_argument(
        '--constraint', '-c',
        type=str,
        nargs='*',
        default=list(CONSTRAINT_CONFIGS.keys()),
        choices=list(CONSTRAINT_CONFIGS.keys()),
        help='3D constraint(s) to learn (sphere, spiral, cylinder, hyperplane). If not specified, runs all constraints.'
    )
    
    parser.add_argument(
        '--archs', '-a',
        type=str,
        nargs='+',
        choices=list(ARCHITECTURES.keys()),
        help='Architectures to use (default: from constraint config)'
    )
    
    parser.add_argument(
        '--noise', '-n',
        type=float,
        nargs='+',
        help='Noise levels to test (default: [0.03])'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['standard', 'trajectory'],
        default='trajectory',
        help='Dataset type: standard (random samples) or trajectory (connected path on surface)'
    )
    
    parser.add_argument(
        '--traj-num',
        type=int,
        default=35,
        help='Number of trajectories to generate (for trajectory dataset, default: 35)'
    )
    
    parser.add_argument(
        '--traj-length',
        type=float,
        default=4.0,
        help='Total length of each trajectory (for trajectory dataset, default: 4.0)'
    )
    
    parser.add_argument(
        '--traj-points',
        type=int,
        default=25,
        help='Number of points per trajectory (for trajectory dataset, default: 25)'
    )

    parser.add_argument(
        '--cv-folds', '-k',
        type=int,
        default=1,
        help='Number of cross-validation folds (default: 1 disables CV)'
    )

    parser.add_argument(
        '--traj-holdout',
        type=float,
        default=0.0,
        help='Fraction of trajectories to hold out from training (e.g., 0.2 for 80/20). Default: 0.0'
    )

    parser.add_argument(
        '--eval-heldout-trajectories',
        action='store_true',
        help='Also evaluate trained models on the held-out trajectories (classification metrics)'
    )
    
    parser.add_argument(
        '--regularizer', '-r',
        type=str,
        nargs='+',
        choices=['none', 'l2', 'eikonal'],
        default=['eikonal'],
        help='Regularizers to test for Neural Pull Loss (default: [eikonal]). Use multiple to compare: --regularizer none l2 eikonal'
    )
    
    parser.add_argument(
        '--disable-plots',
        type=str,
        nargs='+',
        choices=['curves', 'heatmap', 'confusion', 'gt_samples', 'contours', 'projection', 'slices'],
        help='Plots to disable'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Save plots without displaying them (useful for batch processing)'
    )
    
    args = parser.parse_args()
    
    # Prepare enable_plots dict
    enable_plots = {
        "curves": True,
        "heatmap": False,
        "confusion": False,
        "gt_samples": True,
        "contours": True,
        "projection": True,
        "slices": True,
    }
    
    if args.disable_plots:
        for plot_type in args.disable_plots:
            enable_plots[plot_type] = False
    
    # Support multiple constraints
    constraints = args.constraint if isinstance(args.constraint, list) else [args.constraint]
    
    # Global results accumulator
    global_results = []
    
    # Run experiment for each constraint
    for constraint in constraints:
        run_experiment(
            constraint_name=constraint,
            archs=args.archs,
            noise_levels=args.noise,
            enable_plots=enable_plots,
            dataset_type=args.dataset,
            traj_num=args.traj_num,
            traj_length=args.traj_length,
            traj_points=args.traj_points,
            show_plots=not args.no_show,
            global_results=global_results,
            regularizers=args.regularizer,
            cv_folds=args.cv_folds,
            traj_holdout=args.traj_holdout,
            eval_heldout=args.eval_heldout_trajectories
        )
    
    # Save consolidated results if multiple constraints were run
    if len(constraints) > 1 and global_results:
        print(f"\n{'='*80}")
        print("GENERATING CONSOLIDATED COMPARISON ACROSS ALL CONSTRAINTS")
        print(f"{'='*80}\n")
        
        # Create results summary directory
        summary_dir = os.path.join(HERE, "results_summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Determine dataset type and trajectory number from first result
        dataset_suffix = ""
        if global_results:
            dataset_type_used = global_results[0]['dataset']
            if dataset_type_used == "trajectory":
                # Get traj_num from args (passed to run_experiment)
                dataset_suffix = f"_trajectory_{args.traj_num}"
        
        import csv
        from collections import defaultdict
        
        # Determine experiment type:
        # - Regularizer experiment: multiple regularizers OR testing losses beyond Neural Pull
        # - Loss comparison: multiple different loss functions
        is_regularizer_experiment = args.regularizer is not None and len(args.regularizer) > 1
        is_loss_comparison = len(LOSS_VARIANTS) > 1
        
        regularizer_suffix = ""
        if is_regularizer_experiment:
            regularizer_suffix = "_regularizers"
        
        # Save detailed consolidated results
        consolidated_path = os.path.join(summary_dir, f"consolidated_results{dataset_suffix}{regularizer_suffix}.csv")
        with open(consolidated_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Add Regularizer column if it's a regularizer experiment
            if is_regularizer_experiment:
                writer.writerow(['Constraint', 'Architecture', 'Loss', 'Regularizer', 'Noise', 'Dataset',
                               'Chamfer', 'Accuracy(%)', 'Pos_Acc(%)', 'Neg_Acc(%)',
                               'Train_Loss', 'Test_Loss'])
            else:
                writer.writerow(['Constraint', 'Architecture', 'Loss', 'Noise', 'Dataset',
                               'Chamfer', 'Accuracy(%)', 'Pos_Acc(%)', 'Neg_Acc(%)',
                               'Train_Loss', 'Test_Loss'])
            
            for res in global_results:
                if is_regularizer_experiment:
                    writer.writerow([
                        res['constraint'], res['architecture'], res['loss'],
                        res.get('regularizer', 'none'),
                        res['noise'], res['dataset'],
                        f"{res['chamfer']:.6f}",
                        f"{res['train_loss']:.6f}",
                        f"{res['test_loss']:.6f}"
                    ])
                else:
                    writer.writerow([
                        res['constraint'], res['architecture'], res['loss'], 
                        res['noise'], res['dataset'],
                        f"{res['chamfer']:.6f}",
                        f"{res['train_loss']:.6f}",
                        f"{res['test_loss']:.6f}"
                    ])
        
        print(f"✓ Consolidated results saved to: {consolidated_path}")
        
        # If this is a regularizer experiment, also create regularizer comparison
        if is_regularizer_experiment:
            print(f"\n📊 REGULARIZER COMPARISON ACROSS ALL CONSTRAINTS:\n")
            
            # Group results by regularizer (use 'regularizer' field from global_results)
            regularizer_stats = defaultdict(lambda: {'chamfer': [], 'train_loss': [], 'test_loss': [], 'constraints': set()})
            
            for res in global_results:
                # Use the regularizer field directly (only process Neural Pull losses)
                if res['loss'] != 'Neural Pull':
                    continue  # Skip non-Neural Pull losses
                
                reg_type = res.get('regularizer', 'none')
                
                regularizer_stats[reg_type]['chamfer'].append(res['chamfer'])
                regularizer_stats[reg_type]['train_loss'].append(res['train_loss'])
                regularizer_stats[reg_type]['test_loss'].append(res['test_loss'])
                regularizer_stats[reg_type]['constraints'].add(res['constraint'])
            
            # Print summary to console
            print(f"  {'Regularizer':<15} {'Avg Chamfer':<15} {'Avg Acc(%)':<12} {'Avg Pos_Acc(%)':<15} {'Avg Neg_Acc(%)':<15}")
            print(f"  {'-'*70}")
            for reg_name in ['none', 'l2', 'eikonal']:
                if reg_name in regularizer_stats:
                    stats = regularizer_stats[reg_name]
                    avg_chamfer = np.mean(stats['chamfer'])
    
                    print(f"  {reg_name:<15} {avg_chamfer:<15.6f}")
            
            # Save regularizer comparison to CSV
            regularizer_comparison_path = os.path.join(summary_dir, f"regularizer_comparison{dataset_suffix}.csv")
            with open(regularizer_comparison_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Regularizer', 'Num_Constraints', 'Avg_Chamfer', 'Std_Chamfer', 'Min_Chamfer', 'Max_Chamfer',
                                'Avg_Train_Loss', 'Avg_Test_Loss'])
                
                for reg_name in ['none', 'l2', 'eikonal']:
                    if reg_name in regularizer_stats:
                        stats = regularizer_stats[reg_name]
                        writer.writerow([
                            reg_name,
                            len(stats['constraints']),
                            f"{np.mean(stats['chamfer']):.6f}",
                            f"{np.std(stats['chamfer']):.6f}",
                            f"{np.min(stats['chamfer']):.6f}",
                            f"{np.max(stats['chamfer']):.6f}",
                            f"{np.mean(stats['train_loss']):.6f}",
                            f"{np.mean(stats['test_loss']):.6f}"
                        ])
            
            print(f"✓ Regularizer comparison saved to: {regularizer_comparison_path}\n")
        
        # Loss comparison summary (only if comparing different losses)
        if is_loss_comparison:
            print(f"\n📊 SUMMARY BY LOSS FUNCTION:\n")
            # Compute average metrics per loss across all constraints
            loss_stats = defaultdict(lambda: {'chamfer': [], 'acc': [], 'train_loss': [], 'test_loss': []})
            
            for res in global_results:
                loss_stats[res['loss']]['chamfer'].append(res['chamfer'])
                loss_stats[res['loss']]['acc'].append(res.get('accuracy', 0))
                loss_stats[res['loss']]['train_loss'].append(res['train_loss'])
                loss_stats[res['loss']]['test_loss'].append(res['test_loss'])
            
            # Print summary to console
            for loss_name in sorted(loss_stats.keys()):
                avg_chamfer = np.mean(loss_stats[loss_name]['chamfer'])
                avg_acc = np.mean(loss_stats[loss_name]['acc']) * 100
                print(f"  {loss_name:20s}: Chamfer={avg_chamfer:.6f}, Acc={avg_acc:.2f}%")
            
            # Save average summary to CSV
            summary_path = os.path.join(summary_dir, f"loss_averages{dataset_suffix}{regularizer_suffix}.csv")
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Loss', 'Avg_Chamfer', 'Std_Chamfer', 'Min_Chamfer', 'Max_Chamfer',
                            'Avg_Acc(%)', 'Std_Acc(%)', 'Avg_Train_Loss', 'Avg_Test_Loss'])
                for loss_name in sorted(loss_stats.keys()):
                    avg_chamfer = np.mean(loss_stats[loss_name]['chamfer'])
                    std_chamfer = np.std(loss_stats[loss_name]['chamfer'])
                    min_chamfer = np.min(loss_stats[loss_name]['chamfer'])
                    max_chamfer = np.max(loss_stats[loss_name]['chamfer'])
                    avg_acc = np.mean(loss_stats[loss_name]['acc']) * 100
                    std_acc = np.std(loss_stats[loss_name]['acc']) * 100
                    avg_train = np.mean(loss_stats[loss_name]['train_loss'])
                    avg_test = np.mean(loss_stats[loss_name]['test_loss'])
                    
                    writer.writerow([
                        loss_name,
                        f"{avg_chamfer:.6f}",
                        f"{std_chamfer:.6f}",
                        f"{min_chamfer:.6f}",
                        f"{max_chamfer:.6f}",
                        f"{avg_acc:.2f}",
                        f"{std_acc:.2f}",
                        f"{avg_train:.6f}",
                        f"{avg_test:.6f}"
                    ])
            
            print(f" Loss averages saved to: {summary_path}")
            print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

