"""
Sinusoidal reference tracking experiment for MPC on sphere.

This module implements an experiment where an MPC controller tracks
a sinusoidal reference trajectory on a spherical surface while
maintaining the surface constraint h(x)=0.
"""

import numpy as np
import os
import csv
from typing import Tuple, Dict, Optional
import torch

from src.models.mpc import MPCConfig, MPCController
from src.utils.spherical_trajectory import generate_sinusoidal_reference


def run_sinusoidal_experiment(
    h_fn,
    grad_fn,
    model,
    h_fn_learned,
    grad_fn_learned,
    output_dir: str,
    R: float = 1.0,
    N_ref: int = 100,
    amplitude: Optional[float] = None,
    freq: int = 3,
    seed: int = 42,
    show_plots: bool = True
) -> Dict:
    """Run sinusoidal reference tracking experiment.
    
    Args:
        h_fn: Implicit surface function h(x)=0
        grad_fn: Gradient of surface function
        output_dir: Directory to save results
        R: Sphere radius
        N_ref: Number of reference trajectory points
        amplitude: Sinusoidal amplitude (default: 0.3*R)
        freq: Sinusoidal frequency (cycles)
        seed: Random seed
        show_plots: Whether to display plots
        
    Returns:
        Dict with evaluation metrics
    """
    print('\n' + '='*70)
    print('SINUSOIDAL REFERENCE TRACKING EXPERIMENT')
    print('='*70)
    
    if amplitude is None:
        amplitude = 0.3 * R
    
    np.random.seed(seed)

    theta_start = np.random.uniform(0, 2 * np.pi)

 
    offset = np.random.uniform(np.pi/2, 3*np.pi/2)
    theta_goal = theta_start + offset

    
    start = np.array([
        R * np.cos(theta_start), 
        R * np.sin(theta_start), 
        0.0
    ])

    goal = np.array([
        R * np.cos(theta_goal), 
        R * np.sin(theta_goal), 
        0.0
    ])

    print(f'\nSphere radius: {R}')
    print(f'Start point: {start}')
    print(f'Goal point:  {goal}')
    print(f'Start norm: {np.linalg.norm(start):.6f}')
    print(f'Goal norm:  {np.linalg.norm(goal):.6f}')
    
    # Generate sinusoidal reference trajectory (PLANAR VERTICAL - oscillates in Z)
    print(f'\nGenerating PLANAR VERTICAL sinusoidal reference:')
    print(f'  N points: {N_ref}')
    print(f'  Amplitude (Z oscillation): {amplitude:.3f}')
    print(f'  Frequency: {freq} cycles')
    print(f'  NOTE: Reference oscillates in Z axis, MPC must track it while staying on sphere!')
    
    ref_traj = generate_sinusoidal_reference(start, goal, R, N_ref, amplitude, freq)
    
    # Verify reference oscillates in Z
    z_vals = ref_traj[:, 2]
    h_vals = np.array([h_fn(p.reshape(1, -1))[0] for p in ref_traj])
    print(f'\nPlanar vertical reference trajectory verification:')
    print(f'  Z range: [{np.min(z_vals):.3f}, {np.max(z_vals):.3f}]')
    print(f'  Z amplitude: {(np.max(z_vals) - np.min(z_vals))/2:.3f}')
    print(f'  Max |h(x)| (distance to sphere): {np.max(np.abs(h_vals)):.6f}')
    print(f'  Mean |h(x)|: {np.mean(np.abs(h_vals)):.6f}')
    
    # Configure MPC with improved parameters for better tracking
    mpc_config = MPCConfig(
        N=25,                                    # Increased horizon for better prediction
        dt=1.0,
        Q=np.diag([500.0, 500.0, 500.0]),      # Higher weight on tracking error
        R=np.diag([0.1, 0.1, 0.1]),            # Lower weight on control effort
        u_min=np.array([-2.0, -2.0, -2.0]),    # Larger control limits
        u_max=np.array([2.0, 2.0, 2.0]),
        max_sqp_iters=10,                       # More SQP iterations for convergence
        use_augmented_lagrangian=True
    )
    
    print(f'\n--- Running MPC to track reference trajectory ---')
    print(f'MPC Config: N={mpc_config.N}, Q_diag={np.diag(mpc_config.Q)[0]}, R_diag={np.diag(mpc_config.R)[0]}')
    
    mpc = MPCController(config=mpc_config)
    
    # Simulate MPC tracking
    x_mpc = [start.copy()]
    x_current = start.copy()
    
    for step in range(N_ref - 1):
        # Get reference segment for MPC horizon
        horizon_end = min(step + mpc_config.N + 1, N_ref)
        ref_segment = ref_traj[step:horizon_end]
        
        # Current goal is last point in reference segment
        x_goal_step = ref_segment[-1]
        
        # Pass reference segment as waypoints for trajectory tracking
        u_seq, x_seq, info = mpc.solve(
            x0=x_current,
            x_goal=x_goal_step,
            waypoints=list(ref_segment),
            grad_h_fn=grad_fn_learned,
            h_fn=h_fn_learned
        )
        
        # Apply first control
        u_applied = u_seq[0]
        x_next = x_current + u_applied * mpc_config.dt
        
        # Project onto surface
        x_next_tensor = torch.tensor(x_next.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            h_val = float(model(x_next_tensor).item())
        if abs(h_val) > 1e-6:
            with torch.no_grad():
                grad_result = grad_fn_learned(x_next_tensor)
                # Check if result is tensor or numpy array
                if isinstance(grad_result, torch.Tensor):
                    grad = grad_result.cpu().numpy().flatten()
                else:
                    grad = np.asarray(grad_result).flatten()
            x_next = x_next - h_val * grad / np.linalg.norm(grad)**2
        
        x_mpc.append(x_next.copy())
        x_current = x_next
        
        if (step + 1) % 10 == 0:
            print(f'  Step {step+1}/{N_ref-1}')
    
    x_mpc = np.array(x_mpc)
    
    # Evaluate performance
    metrics = evaluate_tracking_performance(
        x_mpc, ref_traj, h_fn, R, output_dir
    )
    
    # Visualize results
    if show_plots:
        visualize_results(
            x_mpc, ref_traj, start, goal, R, 
            metrics, freq, amplitude, output_dir
        )
    
    print('\n' + '='*70)
    print('SINUSOIDAL EXPERIMENT COMPLETED')
    print('='*70)
    
    return metrics


def evaluate_tracking_performance(
    x_mpc: np.ndarray,
    ref_traj: np.ndarray,
    h_fn,
    R: float,
    output_dir: str
) -> Dict:
    """Evaluate MPC tracking performance with proper comparison between planar reference and sphere trajectory."""
    
    # Project planar reference to sphere (maintain X, Y, adjust Z to be on sphere)
    ref_projected = []
    for pt in ref_traj:
        x, y, z_planar = pt
        
        # For a sphere: x² + y² + z² = R²
        # Given x, y, solve for z: z = ±√(R² - x² - y²)
        r_xy_sq = x**2 + y**2
        
        if r_xy_sq >= R**2:
            # Point is outside sphere's XY projection, use radial projection
            norm = np.linalg.norm(pt)
            if norm < 1e-8:
                pt_proj = np.array([R, 0.0, 0.0])
            else:
                pt_proj = (pt / norm) * R
        else:
            # Project vertically: keep X, Y, adjust Z
            z_sphere_sq = R**2 - r_xy_sq
            z_sphere = np.sqrt(z_sphere_sq)
            
            # Choose sign of z that's closer to original z_planar
            if abs(z_planar - z_sphere) < abs(z_planar - (-z_sphere)):
                pt_proj = np.array([x, y, z_sphere])
            else:
                pt_proj = np.array([x, y, -z_sphere])
        
        ref_projected.append(pt_proj)
    ref_projected = np.array(ref_projected)
    
    # Project MPC trajectory to planar space (set to XY plane at average Z of reference)
    avg_z_ref = np.mean(ref_traj[:, 2])
    mpc_projected_to_plane = x_mpc.copy()
    # Keep XY, but project Z to planar reference for fair comparison
    
    # Metric 1: Distance between MPC (on sphere) and reference projected to sphere
    # This shows how well MPC follows the "ideal" spherical trajectory
    tracking_errors_sphere = np.linalg.norm(x_mpc - ref_projected, axis=1)
    
    # Metric 2: Distance between MPC projected to XY plane and planar reference XY
    # This shows how well MPC follows the planar path in XY
    tracking_errors_xy = np.linalg.norm(x_mpc[:, :2] - ref_traj[:, :2], axis=1)
    
    # Metric 3: Z-difference (vertical tracking)
    z_errors = np.abs(x_mpc[:, 2] - ref_traj[:, 2])
    
    # Calculate path lengths
    ref_path_length = np.sum(np.linalg.norm(np.diff(ref_traj, axis=0), axis=1))
    ref_proj_path_length = np.sum(np.linalg.norm(np.diff(ref_projected, axis=0), axis=1))
    mpc_path_length = np.sum(np.linalg.norm(np.diff(x_mpc, axis=0), axis=1))
    
    # Distance from MPC trajectory to sphere surface
    mpc_norms = np.linalg.norm(x_mpc, axis=1)
    surface_distance = np.abs(mpc_norms - R)
    
    print(f'\n{"="*70}')
    print('TRACKING PERFORMANCE ANALYSIS')
    print(f'{"="*70}')
    print(f'\n1. MPC vs Reference Projected to Sphere (3D tracking on sphere):')
    print(f'   Mean error:     {np.mean(tracking_errors_sphere):.6f}')
    print(f'   Max error:      {np.max(tracking_errors_sphere):.6f}')
    print(f'   Std error:      {np.std(tracking_errors_sphere):.6f}')
    
    print(f'\n2. MPC vs Planar Reference in XY plane:')
    print(f'   Mean XY error:  {np.mean(tracking_errors_xy):.6f}')
    print(f'   Max XY error:   {np.max(tracking_errors_xy):.6f}')
    
    print(f'\n3. Vertical (Z) tracking:')
    print(f'   Mean Z error:   {np.mean(z_errors):.6f}')
    print(f'   Max Z error:    {np.max(z_errors):.6f}')
    
    print(f'\n4. Path lengths:')
    print(f'   Planar reference:      {ref_path_length:.6f}')
    print(f'   Reference on sphere:   {ref_proj_path_length:.6f}')
    print(f'   MPC on sphere:         {mpc_path_length:.6f}')
    
    print(f'\n5. Constraint satisfaction (distance to sphere):')
    print(f'   Mean:   {np.mean(surface_distance):.6e}')
    print(f'   Max:    {np.max(surface_distance):.6e}')
 
    # Save metrics
    metrics_path = os.path.join(output_dir, 'sinusoidal_metrics.csv')
    metrics = {
        'mean_error_sphere': np.mean(tracking_errors_sphere),
        'max_error_sphere': np.max(tracking_errors_sphere),
        'mean_error_xy': np.mean(tracking_errors_xy),
        'max_error_xy': np.max(tracking_errors_xy),
        'mean_error_z': np.mean(z_errors),
        'max_error_z': np.max(z_errors),
        'ref_path_length': ref_path_length,
        'ref_proj_path_length': ref_proj_path_length,
        'mpc_path_length': mpc_path_length,
        'mean_surface_dist': np.mean(surface_distance),
        'max_surface_dist': np.max(surface_distance),
        'ref_projected': ref_projected  # Include projected reference for visualization
    }
    
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Mean Error (MPC vs Ref on Sphere)', f'{metrics["mean_error_sphere"]:.6f}'])
        writer.writerow(['Max Error (MPC vs Ref on Sphere)', f'{metrics["max_error_sphere"]:.6f}'])
        writer.writerow(['Mean XY Error', f'{metrics["mean_error_xy"]:.6f}'])
        writer.writerow(['Max XY Error', f'{metrics["max_error_xy"]:.6f}'])
        writer.writerow(['Mean Z Error', f'{metrics["mean_error_z"]:.6f}'])
        writer.writerow(['Max Z Error', f'{metrics["max_error_z"]:.6f}'])
        writer.writerow(['Planar Reference Path Length', f'{metrics["ref_path_length"]:.6f}'])
        writer.writerow(['Projected Reference Path Length', f'{metrics["ref_proj_path_length"]:.6f}'])
        writer.writerow(['MPC Path Length', f'{metrics["mpc_path_length"]:.6f}'])
        writer.writerow(['Mean Surface Distance', f'{metrics["mean_surface_dist"]:.6e}'])
        writer.writerow(['Max Surface Distance', f'{metrics["max_surface_dist"]:.6e}'])
     
    
    print(f'\n--- Metrics saved to: {metrics_path} ---')
    
    return metrics


def visualize_results(
    x_mpc: np.ndarray,
    ref_traj: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    R: float,
    metrics: Dict,
    freq: int,
    amplitude: float,
    output_dir: str
):
    """Create visualization: planar reference + MPC on sphere."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sphere constraint function for surface visualization
    def h_sphere(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.sum(x**2, axis=1) - R**2
    
    # Draw sphere surface
    from src.utils.viz_cont import compute_isosurface
    max_bound = 1.5
    sphere_bounds = [-max_bound, max_bound, -max_bound, max_bound, -max_bound, max_bound]
    verts, faces = compute_isosurface(h_sphere, bounds=sphere_bounds, res=48)
    if verts is not None and faces is not None:
        try:
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, 
                          alpha=0.2, color='lightcoral')
        except Exception:
            pass
    
    # Only 2 trajectories:
    # 1. Planar Reference (blue, solid) - the sinusoidal reference in the plane
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 
           '-', color='#1f77b4', linewidth=3.5, alpha=0.8, label='Planar Reference')
    
    # 2. MPC Tracking (green, dashed) - on the sphere following the reference
    ax.plot(x_mpc[:, 0], x_mpc[:, 1], x_mpc[:, 2], 
           '--', color='#2ca02c', linewidth=3.5, alpha=0.9, label='MPC on Sphere')
    
    # Add start/end markers
    # Planar reference markers
    ax.scatter(ref_traj[0, 0], ref_traj[0, 1], ref_traj[0, 2], 
              s=150, marker='o', color='#1f77b4', edgecolors='black', linewidths=2)
    ax.scatter(ref_traj[-1, 0], ref_traj[-1, 1], ref_traj[-1, 2], 
              s=150, marker='s', color='#1f77b4', edgecolors='black', linewidths=2)
    
    # MPC markers
    ax.scatter(x_mpc[0, 0], x_mpc[0, 1], x_mpc[0, 2], 
              s=150, marker='o', color='#2ca02c', edgecolors='black', linewidths=2)
    ax.scatter(x_mpc[-1, 0], x_mpc[-1, 1], x_mpc[-1, 2], 
              s=150, marker='s', color='#2ca02c', edgecolors='black', linewidths=2)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=16, labelpad=10)
    ax.set_ylabel('Y', fontsize=16, labelpad=10)
    ax.set_zlabel('Z', fontsize=16, labelpad=10)
    ax.set_title('Planar Reference vs MPC Tracking on Sphere', fontsize=18)
    ax.legend(fontsize=14, loc='upper right')
    ax.view_init(elev=35, azim=45)
    ax.grid(True)
    
    # Set aspect ratio
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
    except Exception:
        pass
    
    plt.tight_layout()
    
    # Save static figure
    save_path = os.path.join(output_dir, 'sinusoidal_tracking_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'\n--- Static visualization saved to: {save_path} ---')
    
    plt.show()
    plt.close()
    
    # Create animated version
    print('\n--- Generating animated visualization ---')
    create_animated_tracking(x_mpc, ref_traj, R, output_dir)


def create_animated_tracking(
    x_mpc: np.ndarray,
    ref_traj: np.ndarray,
    R: float,
    output_dir: str
):
    """Create animated visualization: static planar reference + animated MPC on sphere."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create sphere constraint function
    def h_sphere(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.sum(x**2, axis=1) - R**2
    
    # Draw sphere surface (static)
    from src.utils.viz_cont import compute_isosurface
    max_bound = 1.5
    sphere_bounds = [-max_bound, max_bound, -max_bound, max_bound, -max_bound, max_bound]
    verts, faces = compute_isosurface(h_sphere, bounds=sphere_bounds, res=48)
    if verts is not None and faces is not None:
        try:
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, 
                          alpha=0.15, color='lightcoral')
        except Exception:
            pass
    
    # Plot planar reference (static, solid blue)
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 
           '-', color='#1f77b4', linewidth=3.0, alpha=0.7, label='Planar Reference')
    
    # Initialize MPC trajectory (will be updated in animation)
    mpc_line, = ax.plot([], [], [], '--', color='#2ca02c', linewidth=3.5, alpha=0.9, label='MPC on Sphere')
    mpc_marker = ax.scatter([], [], [], s=200, marker='o', color='#2ca02c', edgecolors='black', linewidths=2)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=16, labelpad=10)
    ax.set_ylabel('Y', fontsize=16, labelpad=10)
    ax.set_zlabel('Z', fontsize=16, labelpad=10)
    ax.set_title('MPC Tracking Planar Reference on Sphere', fontsize=18)
    ax.legend(fontsize=14, loc='upper right')
    ax.view_init(elev=20, azim=135)  # Initial view
    ax.grid(True)
    
    # Set fixed axis limits
    ax.set_xlim([-max_bound, max_bound])
    ax.set_ylim([-max_bound, max_bound])
    ax.set_zlim([-max_bound, max_bound])
    
    # Set aspect ratio
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    
    plt.tight_layout()
    
    # Animation update function with rotation
    def update(frame):
        # Update MPC trajectory up to current frame
        mpc_line.set_data(x_mpc[:frame, 0], x_mpc[:frame, 1])
        mpc_line.set_3d_properties(x_mpc[:frame, 2])
        
        # Update current position marker
        if frame > 0:
            mpc_marker._offsets3d = ([x_mpc[frame-1, 0]], [x_mpc[frame-1, 1]], [x_mpc[frame-1, 2]])
        
        # Rotate view during animation
        total_frames = len(frames)
        frame_idx = list(frames).index(frame) if frame in frames else 0
        azim = 135 + (frame_idx / total_frames) * 360  # Full 360 degree rotation
        ax.view_init(elev=20, azim=azim)
        
        return mpc_line, mpc_marker
    
    # Create animation
    num_frames = len(x_mpc)
    frame_step = max(1, num_frames // 60)  # 60 frames for smooth animation
    frames = range(1, num_frames + 1, frame_step)
    
    anim = FuncAnimation(fig, update, frames=frames, interval=150, blit=False, repeat=True)
    
    # Save as GIF
    save_path_gif = os.path.join(output_dir, 'sinusoidal_tracking_animated.gif')
    writer = PillowWriter(fps=7)  # 7 fps - velocitat intermèdia
    anim.save(save_path_gif, writer=writer, dpi=100)
    
    print(f'--- Animated visualization saved to: {save_path_gif} ---')
    
    plt.close()


__all__ = ['run_sinusoidal_experiment']
