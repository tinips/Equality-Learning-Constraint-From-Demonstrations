"""
Simple PCA Negative Sampling Quality Comparison

Usage:
    python main_simple.py           # Compare all 5 strategies
    python main_simple.py --ref     # Compare only Adaptive PCA vs Reference PCA
"""

import os
import sys
import csv
import argparse
import numpy as np

# Setup paths
HERE = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets3D import (
    make_constraint_3d,
    generate_surface_trajectory,
    sample_negative_from_estimated_normals,
    sample_negative_from_normals_3d,
    sample_negative_random_direction,
    sample_negative_reference_pca,
    estimate_local_normals_adaptive,
    estimate_local_normals
)
from configs import config3D as config
from src.utils.seed import set_seed
from src.utils.metrics import chamfer_distance_symmetric, compute_chamfer_distance
from src.models.models3D import SpiralImplicitMLP
from src.training.train_loop import train_one_with_split
from src.models.ecomann import ECoMaNN, train_ecomann, train_ecomann_with_negatives
from src.utils.viz3D import plot_GT_LS_Countours, plot_GT_with_samples_grid
import torch

# Constants
CONSTRAINTS = ['sphere', 'ellipsoid', 'cylinder']
NOISE_LEVEL = 0.03  # Noise on positives
NEG_STD = 0.30      # Constant displacement magnitude
CLOSE_THRESHOLD = 0.29  # Threshold < displacement (0.29 < 0.30 for numerical precision)
TRAJ_NUM = 35
TRAJ_POINTS = 25
TRAJ_LENGTH = 4.0

# Training constants
EPOCHS = 600
LEARNING_RATE = 0.001
LOSS_TYPE = 6  # Neural Pull Loss (ours)

# ECoMaNN paper loss weights (exact paper values)
LAMBDA_NORM = 1.0
LAMBDA_REFLECTION = 1.0
LAMBDA_FRACTION = 1.0
LAMBDA_SUBSPACE = 1.0
ECOMANN_EPOCHS = 25  # Paper: 25 for clean data


def train_model_with_data(pos_samples, neg_samples, loss_method='neural_pull'):
    """
    Train a model with given positive and negative samples.
    
    Args:
        pos_samples: Positive samples (numpy array)
        neg_samples: Negative samples (numpy array)
        loss_method: 'neural_pull' (ours) or 'ecomann' (reference paper)
        
    Returns:
        model: Trained PyTorch model
    """
    if loss_method == 'ecomann':
        # Train with ECoMaNN (exact paper implementation)
        # Estimate normals using Adaptive PCA
        pca_normals = estimate_local_normals_adaptive(pos_samples, k_neighbors=None, verbose=False)
        
        # Train ECoMaNN with paper-exact method
        model, train_losses = train_ecomann_with_negatives(
            pos_samples=pos_samples,
            neg_samples=neg_samples,
            pca_normals=pca_normals,
            epochs=ECOMANN_EPOCHS,  # Paper: 25 epochs
            lr=LEARNING_RATE,
            batch_size=128,
            lambda_norm=LAMBDA_NORM,
            lambda_reflection=LAMBDA_REFLECTION,
            lambda_fraction=LAMBDA_FRACTION,
            lambda_subspace=LAMBDA_SUBSPACE,
            hidden_sizes=None,  # Use paper default [36, 24, 18, 10]
            verbose=False
        )
        
        return model
    else:
        # Train with Neural Pull Loss (our method)
        # Convert to tensors
        pos_tensor = torch.from_numpy(pos_samples).float()
        neg_tensor = torch.from_numpy(neg_samples).float()
        
        # Create 80/20 train/eval split
        n_pos = len(pos_tensor)
        n_neg = len(neg_tensor)
        n_pos_train = int(0.8 * n_pos)
        n_neg_train = int(0.8 * n_neg)
        
        # Split data
        pos_train = pos_tensor[:n_pos_train]
        pos_eval = pos_tensor[n_pos_train:]
        neg_train = neg_tensor[:n_neg_train]
        neg_eval = neg_tensor[n_neg_train:]
        
        # Create model
        model = SpiralImplicitMLP()
        
        # Train with Neural Pull Loss + Eikonal regularizer
        result = train_one_with_split(
            model=model,
            pos_train=pos_train,
            neg_train=neg_train,
            pos_eval=pos_eval,
            neg_eval=neg_eval,
            epochs=EPOCHS,
            variant=LOSS_TYPE,
            lr=LEARNING_RATE,
            batch=128,
            regularizer='eikonal'
        )
        
        # Unpack result (variant 6 returns 4 values, others return 3)
        if LOSS_TYPE == 6:
            trained_model, train_losses, test_losses, projected_info = result
        else:
            trained_model, train_losses, test_losses = result
        
        return trained_model


def evaluate_learned_surface(model, gt_samples_fn, constraint_name):
    """
    Evaluate learned surface by computing Chamfer distance to ground truth.
    
    Args:
        model: Trained model
        gt_samples_fn: Function to sample ground truth surface
        constraint_name: Name of constraint
        
    Returns:
        chamfer_dist: Chamfer distance between learned and GT surface
    """
    # Set bounds based on constraint
    bounds_config = {
        'sphere': [-1.3, 1.3, -1.3, 1.3, -1.3, 1.3],
        'ellipsoid': [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0],
        'cylinder': [-1.3, 1.3, -1.3, 1.3, -2.2, 2.2]
    }
    bounds = bounds_config.get(constraint_name, [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0])
    
    chamfer_dist = compute_chamfer_distance(
        model=model,
        gt_samples_fn=gt_samples_fn,
        level=0.0,
        bounds=bounds,
        resolution=50,
        n_samples=5000,
        seed=42
    )
    
    return chamfer_dist


def generate_trajectories(constraint_name, num_traj, num_points, length):
    """Generate trajectories on surface"""
    h_np, grad_np, gt_samples, _ = make_constraint_3d(constraint_name)
    
    trajectories = []
    for i in range(num_traj):
        traj_seed = int(config.SEED) + i + 1
        traj_points = generate_surface_trajectory(
            total_length=length,
            num_steps=num_points,
            h=h_np,
            grad=grad_np,
            get_samples=gt_samples,
            seed=traj_seed
        )
        if len(traj_points) > 0:
            trajectories.append(traj_points)
    
    # Concatenate all trajectories
    pos_samples = np.concatenate(trajectories, axis=0)
    return pos_samples, h_np, grad_np, gt_samples


def evaluate_negative_quality(constraint_name, ref_only=False, train_models=False, loss_method='neural_pull', compare_methods_only=False):
    """
    Evaluate quality of negative samples generated by different strategies
    
    Args:
        constraint_name: Name of the constraint (sphere, ellipsoid, cylinder)
        ref_only: If True, only compare Adaptive PCA vs Reference PCA
        train_models: If True, also train models and compute learned surface Chamfer
        loss_method: 'neural_pull', 'ecomann', or 'both'
        compare_methods_only: If True, only compare loss methods with Adaptive PCA (skip negative quality comparison)
    
    Returns:
        dict: Quality metrics for each strategy (empty if compare_methods_only=True)
        dict: (if train_models=True) Chamfer distances of learned surfaces
    """
    print(f"\n{'='*70}")
    print(f"Constraint: {constraint_name.upper()}")
    print(f"{'='*70}")
    
    # Generate trajectories
    print(f"Generating {TRAJ_NUM} trajectories...")
    pos_clean, h_np, grad_np, gt_samples = generate_trajectories(
        constraint_name, TRAJ_NUM, TRAJ_POINTS, TRAJ_LENGTH
    )
    print(f"  ✓ {len(pos_clean)} trajectory points")
    
    # Add noise to positives
    pos_noisy = pos_clean + np.random.normal(0.0, NOISE_LEVEL, size=pos_clean.shape)
    print(f"  ✓ Added noise (std={NOISE_LEVEL})")
    
    # Generate negatives with different strategies
    # Use CONSTANT displacement (not random) to fairly compare normal estimation quality
    # If normal direction is good, negatives will be far from surface
    # If normal direction is bad, some negatives will be close to surface
    
    if compare_methods_only:
        # Only compare loss methods with Adaptive PCA negatives
        print(f"\n  Comparing loss methods with Adaptive PCA negatives...")
        print(f"     Using constant displacement: {NEG_STD}")
        
        strategies = [
            ('Adaptive PCA', sample_negative_from_estimated_normals(
                pos_noisy, std=NEG_STD, use_adaptive=True, use_constant_displacement=True
            ))
        ]
    elif ref_only:
        print(f"\n  Generating negatives (Reference PCA vs Adaptive PCA)...")
        print(f"     Using constant displacement: {NEG_STD}")
        
        strategies = [
            ('Adaptive PCA', sample_negative_from_estimated_normals(
                pos_noisy, std=NEG_STD, use_adaptive=True, use_constant_displacement=True
            )),
            ('Reference PCA', sample_negative_reference_pca(
                pos_noisy, std=NEG_STD, k_neighbors=10, use_constant_displacement=True
            ))
        ]
    else:
        print(f"\n  Generating negatives (all strategies)...")
        print(f"     Using constant displacement: {NEG_STD}")
        
        strategies = [
            ('Random Direction', sample_negative_random_direction(
                pos_noisy, std=NEG_STD, use_constant_displacement=True
            )),
            ('PCA Normals (k=10)', sample_negative_from_estimated_normals(
                pos_noisy, std=NEG_STD, k_neighbors=10, use_adaptive=False, use_constant_displacement=True
            )),
            ('Adaptive PCA', sample_negative_from_estimated_normals(
                pos_noisy, std=NEG_STD, use_adaptive=True, use_constant_displacement=True
            )),
            ('Reference PCA', sample_negative_reference_pca(
                pos_noisy, std=NEG_STD, k_neighbors=10, use_constant_displacement=True
            )),
            # GT normals use CLEAN positions (without noise) for perfect normals
            ('Ground Truth Normals', sample_negative_from_normals_3d(
                pos_clean, grad_np, h_np, std=NEG_STD, use_constant_displacement=True
            ))
        ]
    
    # Generate reference surface samples for chamfer distance computation (only if evaluating quality)
    results = {}
    learned_surface_results = {} if train_models else None
    
    if not compare_methods_only:
        n_ref_samples = 2000
        ref_surface_samples = gt_samples(n_ref_samples)
        
        # Evaluate quality
        print(f"\n  📏 Evaluating negative sample quality:")
        print(f"     Threshold for 'too close': {CLOSE_THRESHOLD} (< displacement {NEG_STD} for numerical stability)")
        print(f"     → GT normals should give ~0% (perfect perpendicular)")
        print(f"     → Bad normals will have some samples < {CLOSE_THRESHOLD}")
        print()
    
    for strategy_name, neg_samples in strategies:
        # Compute distances from GT surface (only if not comparing methods only)
        if not compare_methods_only:
            distances = np.abs(h_np(neg_samples))
            
            # Count "too close" samples
            too_close = np.sum(distances < CLOSE_THRESHOLD)
            too_close_pct = 100.0 * too_close / len(neg_samples)
            
            # Compute chamfer distance (negatives vs surface)
            chamfer_dist = chamfer_distance_symmetric(neg_samples, ref_surface_samples)
            
            # Statistics
            results[strategy_name] = {
                'n_samples': len(neg_samples),
                'min_dist': np.min(distances),
                'max_dist': np.max(distances),
                'mean_dist': np.mean(distances),
                'median_dist': np.median(distances),
                'too_close': too_close,
                'too_close_pct': too_close_pct,
                'chamfer_distance': chamfer_dist
            }
            
            print(f"  {strategy_name}:")
            print(f"    Distance → Min: {results[strategy_name]['min_dist']:.4f}  Max: {results[strategy_name]['max_dist']:.4f}  Mean: {results[strategy_name]['mean_dist']:.4f}")
            print(f"    Too close: {too_close}/{len(neg_samples)} ({too_close_pct:.1f}%)")
            print(f"    Neg-Surface Chamfer: {chamfer_dist:.4f}")
        else:
            print(f"  Using {strategy_name} for negative sampling")
        
        # Train model and evaluate learned surface if requested
        if train_models:
            # Determine which methods to train
            methods_to_train = []
            if loss_method == 'both':
                methods_to_train = [('neural_pull', 'Neural Pull'), ('ecomann', 'ECoMaNN')]
            else:
                method_name = 'Neural Pull' if loss_method == 'neural_pull' else 'ECoMaNN'
                methods_to_train = [(loss_method, method_name)]
            
            for loss_method, method_display in methods_to_train:
                strategy_full_name = f"{strategy_name} + {method_display}"
                print(f"    Training model with {method_display} ({EPOCHS} epochs)...")
                model = train_model_with_data(pos_noisy, neg_samples, loss_method=loss_method)
                
                print(f"    Evaluating learned surface...")
                learned_chamfer = evaluate_learned_surface(model, gt_samples, constraint_name)
                
                learned_surface_results[strategy_full_name] = {
                    'learned_surface_chamfer': learned_chamfer,
                    'model': model  # Save model for visualization
                }
                
                print(f"    Learned Surface Chamfer: {learned_chamfer:.6f}")
    
    if train_models:
        return results, learned_surface_results
    return results


def plot_learned_surfaces(all_learned_results, output_dir):
    """
    Generate visualizations of learned surfaces for each constraint.
    
    Args:
        all_learned_results: Dictionary with structure {constraint: {strategy: {'model': model, ...}}}
        output_dir: Directory to save plots
    """
    print(f"\n{'='*70}")
    print("GENERATING LEARNED SURFACE VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Bounds configuration for each constraint
    bounds_config = {
        'sphere': [-1.3, 1.3, -1.3, 1.3, -1.3, 1.3],
        'ellipsoid': [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0],
        'cylinder': [-1.3, 1.3, -1.3, 1.3, -2.2, 2.2]
    }
    
    for constraint_name, strategy_results in all_learned_results.items():
        print(f"\n  Plotting {constraint_name}...")
        
        # Get GT function for this constraint
        h_np, grad_np, gt_samples, gt_curve = make_constraint_3d(constraint_name)
        bounds = bounds_config.get(constraint_name, [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0])
        
        # Get some GT samples for visualization
        gt_samples_viz = gt_samples(1000)
        
        # Generate training data for visualization (same as training) - ONLY ONCE per constraint
        pos_clean, _, _, _ = generate_trajectories(constraint_name, TRAJ_NUM, TRAJ_POINTS, TRAJ_LENGTH)
        pos_viz = pos_clean + np.random.normal(0.0, NOISE_LEVEL, size=pos_clean.shape)
        # Use normal Gaussian displacement (like main_3d.py) instead of constant displacement
        neg_viz = sample_negative_from_estimated_normals(
            pos_viz, std=NEG_STD, use_adaptive=True, use_constant_displacement=False
        )
        
        # Plot GT with training samples ONCE per constraint (data is the same for all methods)
        samples_path = os.path.join(output_dir, f"{constraint_name}_training_samples.png")
        plot_GT_with_samples_grid(
            gt_samples=gt_samples_viz,
            pos_np=pos_viz[:],  
            neg_np=neg_viz[:],
            bounds=bounds,
            constraint_name=constraint_name,
            gt_function=h_np,
            save_path=samples_path
        )
        print(f"    ✓ Saved training samples: {samples_path}")
        
        # Plot learned surface contours for each method
        for strategy_name, result_data in strategy_results.items():
            if 'model' not in result_data:
                continue
                
            model = result_data['model']
            
            # Create results dict in expected format: {(arch, loss): (model, train_curve, test_curve)}
            results_for_plot = {
                (constraint_name, strategy_name): (model, [], [])
            }
            
            # Sanitize filename
            safe_strategy = strategy_name.replace(' ', '_').replace('+', 'plus')
            
            # Plot GT + Learned Surface Contours
            contour_path = os.path.join(output_dir, f"{constraint_name}_{safe_strategy}_contours.png")
            plot_GT_LS_Countours(
                results=results_for_plot,
                constraint_name=constraint_name,
                bounds=bounds,
                gt_function=h_np,
                save_path=contour_path,
                level=0.0,
                resolution=30
            )
            print(f"    ✓ Saved contours: {contour_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}")


def save_learned_surface_csv(all_results, output_path):
    """Save learned surface Chamfer distances to CSV file"""
    # Prepare data
    constraints = sorted(all_results.keys())
    all_strategies = set()
    for constraint_results in all_results.values():
        all_strategies.update(constraint_results.keys())
    strategies = sorted(all_strategies)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header - only Chamfer distance
        header = ['Strategy']
        for c in constraints:
            header.append(f"{c} - Chamfer")
        header.append('Avg Chamfer')
        writer.writerow(header)
        
        # Data rows
        for strategy in strategies:
            row = [strategy]
            chamfers = []
            for constraint in constraints:
                if strategy in all_results[constraint]:
                    cd = all_results[constraint][strategy]['learned_surface_chamfer']
                    row.append(f"{cd:.6f}")
                    chamfers.append(cd)
                else:
                    row.append('N/A')
            
            # Average
            if chamfers:
                avg_chamfer = np.mean(chamfers)
                row.append(f"{avg_chamfer:.6f}")
            else:
                row.append('N/A')
            
            writer.writerow(row)
    
    print(f"\n{'='*70}")
    print(f"✓ Learned surface results saved to: {output_path}")
    print(f"{'='*70}")


def save_results_csv(all_results, output_path):
    """Save results to CSV file"""
    # Prepare data
    constraints = sorted(all_results.keys())
    all_strategies = set()
    for constraint_results in all_results.values():
        all_strategies.update(constraint_results.keys())
    strategies = sorted(all_strategies)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header - include Close %, Min dist, Max dist, Chamfer Distance for each constraint
        header = ['Strategy']
        for c in constraints:
            header.extend([f"{c} - Close (%)", f"{c} - Min", f"{c} - Max", f"{c} - Chamfer"])
        writer.writerow(header)
        
        # Data rows
        for strategy in strategies:
            row = [strategy]
            for constraint in constraints:
                if strategy in all_results[constraint]:
                    pct = all_results[constraint][strategy]['too_close_pct']
                    min_dist = all_results[constraint][strategy]['min_dist']
                    max_dist = all_results[constraint][strategy]['max_dist']
                    chamfer = all_results[constraint][strategy]['chamfer_distance']
                    row.extend([f"{pct:.2f}", f"{min_dist:.4f}", f"{max_dist:.4f}", f"{chamfer:.4f}"])
                else:
                    row.extend(['N/A', 'N/A', 'N/A', 'N/A'])
            writer.writerow(row)
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Compare PCA negative sampling quality and loss methods')
    parser.add_argument('--ref', action='store_true', 
                       help='Compare only Reference PCA vs Adaptive PCA (default: all 5 strategies)')
    parser.add_argument('--train', action='store_true',
                       help='Also train models and evaluate learned surfaces (slower)')
    parser.add_argument('--method', type=str, default='neural_pull',
                       choices=['neural_pull', 'ecomann', 'both'],
                       help='Loss method: neural_pull (ours), ecomann (reference paper), or both')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    print("\n" + "="*70)
    print("PCA NEGATIVE SAMPLING & LOSS METHOD COMPARISON")
    print("="*70)
    print(f"Mode: {'Reference PCA vs Adaptive PCA' if args.ref else 'All 5 strategies'}")
    method_display = {'neural_pull': 'Neural Pull (Ours)', 'ecomann': 'ECoMaNN (Reference)', 'both': 'Both Methods'}
    print(f"Loss Method: {method_display[args.method]}")
    print(f"Constraints: {', '.join(CONSTRAINTS)}")
    print(f"Trajectories: {TRAJ_NUM} × {TRAJ_POINTS} points")
    print(f"Noise (positives): {NOISE_LEVEL}")
    print(f"Std (negatives): {NEG_STD}")
    print(f"Threshold (too close): {CLOSE_THRESHOLD}")
    print("="*70)
    
    # Evaluate all constraints
    all_results = {}
    all_learned_results = {} if args.train else None
    
    # Determine if we're comparing methods only (skip negative quality evaluation)
    compare_methods_only = args.method in ['ecomann', 'both']
    
    for constraint in CONSTRAINTS:
        if args.train:
            results, learned_results = evaluate_negative_quality(
                constraint, 
                ref_only=args.ref, 
                train_models=True, 
                loss_method=args.method,
                compare_methods_only=compare_methods_only
            )
            all_results[constraint] = results
            all_learned_results[constraint] = learned_results
        else:
            results = evaluate_negative_quality(
                constraint, 
                ref_only=args.ref, 
                train_models=False,
                compare_methods_only=compare_methods_only
            )
            all_results[constraint] = results
    
    # Save results
    output_dir = os.path.join(HERE, "negative_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create suffix based on mode
    if compare_methods_only:
        suffix = f"_{args.method}"
    else:
        suffix = "_ref" if args.ref else "_all"
    
    # Save negative quality results (skip if comparing methods only)
    if not compare_methods_only:
        output_path = os.path.join(output_dir, f"negative_quality_comparison{suffix}.csv")
        save_results_csv(all_results, output_path)
    
    # Save learned surface results if training was performed
    if args.train:
        learned_output_path = os.path.join(output_dir, f"learned_surface_chamfer{suffix}.csv")
        save_learned_surface_csv(all_learned_results, learned_output_path)
        
        # Generate visualizations of learned surfaces
        plots_dir = os.path.join(output_dir, "learned_surface_plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_learned_surfaces(all_learned_results, plots_dir)
    
    # Print summary (skip if comparing methods only)
    if not compare_methods_only:
        print(f"\n{'='*70}")
        print("SUMMARY: Close Negatives Percentage (Lower is Better)")
        print("="*70)
        print(f"{'Strategy':<25} {' | '.join([f'{c:>12}' for c in CONSTRAINTS])}")
        print("-"*70)
        
        all_strategies = set()
        for constraint_results in all_results.values():
            all_strategies.update(constraint_results.keys())
        
        for strategy in sorted(all_strategies):
            values = []
            for constraint in CONSTRAINTS:
                if strategy in all_results[constraint]:
                    pct = all_results[constraint][strategy]['too_close_pct']
                    values.append(f"{pct:>11.1f}%")
                else:
                    values.append(f"{'N/A':>12}")
            print(f"{strategy:<25} {' | '.join(values)}")
        
        print(f"\n{'='*70}")
        print("SUMMARY: Neg-Surface Chamfer Distance (Higher is Better)")
        print("="*70)
        print(f"{'Strategy':<25} {' | '.join([f'{c:>12}' for c in CONSTRAINTS])}")
        print("-"*70)
        
        for strategy in sorted(all_strategies):
            values = []
            for constraint in CONSTRAINTS:
                if strategy in all_results[constraint]:
                    chamfer = all_results[constraint][strategy]['chamfer_distance']
                    values.append(f"{chamfer:>12.4f}")
                else:
                    values.append(f"{'N/A':>12}")
            print(f"{strategy:<25} {' | '.join(values)}")
    
    # Print learned surface summary if training was performed
    if args.train:
        # Get all strategies from learned results
        all_learned_strategies = set()
        for constraint_results in all_learned_results.values():
            all_learned_strategies.update(constraint_results.keys())
        
        print(f"\n{'='*70}")
        print("SUMMARY: Learned Surface Chamfer Distance (Lower is Better)")
        print("="*70)
        print(f"{'Strategy':<25} {' | '.join([f'{c:>12}' for c in CONSTRAINTS])} | {'Average':>12}")
        print("-"*70)
        
        for strategy in sorted(all_learned_strategies):
            values = []
            chamfers = []
            for constraint in CONSTRAINTS:
                if strategy in all_learned_results[constraint]:
                    cd = all_learned_results[constraint][strategy]['learned_surface_chamfer']
                    values.append(f"{cd:>12.6f}")
                    chamfers.append(cd)
                else:
                    values.append(f"{'N/A':>12}")
            
            # Average
            if chamfers:
                avg = np.mean(chamfers)
                values.append(f"{avg:>12.6f}")
            else:
                values.append(f"{'N/A':>12}")
            
            print(f"{strategy:<25} {' | '.join(values)}")
    
    print("="*70)
    if args.train:
        print(f"✅ Done! Files saved:")
        if compare_methods_only:
            print(f"   - Loss method comparison CSV: {learned_output_path}")
            print(f"   - Learned surface plots: {plots_dir}")
        else:
            print(f"   - Negative quality CSV: {output_path}")
            print(f"   - Learned surfaces CSV: {learned_output_path}")
            print(f"   - Learned surface plots: {plots_dir}")
    else:
        if not compare_methods_only:
            print(f"✅ Done! CSV saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
