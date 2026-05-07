
import numpy as np
import torch
import sys
import os
import argparse
import pickle

# Afegir el directori arrel del projecte al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.surface_utils import load_learned_surface
from src.utils.viz_cont import plot_trajectory_comparison, plot_gt_surface_with_trajectories, plot_projection_comparison, plot_trajectory_comparison_MPC_animated
from src.utils.viz import plot_learning_curves
from src.utils.seed import set_seed
from src.models.controller import ImitationController
from src.models.mpc import MPCConfig, MPCController
from src.training.train_controller import train_controller
from src.training.eval_controller import (
    evaluate_all_test_trajectories,
    generate_trajectory_qp,
    generate_trajectory
)
from src.data.datasets3D import make_constraint_3d, generate_surface_trajectory
from src.data.datasetcontroller import trajectories_to_states_targets
from src.data.obstacle_constraints import make_cylinder_obstacle, make_rectangular_box_obstacle
from src.training.sinusoidal_experiment import run_sinusoidal_experiment




def main(constraint_name, trajectories_path=None, num_epochs=300, seed=42, show_plots=True, qp_mode=False, use_mpc=False, cont=False, use_inequality=False, sinusoidal_mode=False, animated_mpc=False):
    """Load trajectories and train/evaluate controller.
    
    Args:
        constraint_name: Name of constraint ('sphere', 'ellipsoid', 'cylinder', etc.)
        trajectories_path: Path to saved trajectories pickle file
        num_epochs: Training epochs (only used if not in qp_mode)
        seed: Random seed
        show_plots: Whether to display plots
        qp_mode: If True, only evaluate with QP (no training)
        use_inequality: If True, add inequality constraint (obstacle avoidance)
        sinusoidal_mode: If True, run sinusoidal reference tracking experiment
    """
    set_seed(seed)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), constraint_name, 'controller_outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n--- Output directory: {output_dir} ---')
    
    # Determine trajectories path
    if trajectories_path is None:
        trajectories_dir = os.path.join(os.path.dirname(__file__), constraint_name, 'outputs')
        trajectories_train_path = os.path.join(trajectories_dir, 'training_trajectories_traj35_pts25.pkl')
        trajectories_test_path = os.path.join(trajectories_dir, 'testing_trajectories_traj3_pts25.pkl')
    
    
    print(f'\n--- Loading trajectories from: {trajectories_path} ---')
    with open(trajectories_train_path, 'rb') as f:
        trajectories_train = pickle.load(f)
    
    with open(trajectories_test_path, 'rb') as f:
        trajectories_test = pickle.load(f)
    

    states, targets = trajectories_to_states_targets(trajectories_train)
    states_test, targets_test = trajectories_to_states_targets(trajectories_test) 
    
    states_train = states
    targets_train = targets
    h_np, grad_np, gt_samples, gt_curve = make_constraint_3d(constraint_name)

    # Safe wrappers: always provide h_fn/grad_fn based on GT (overridden in QP mode)
    def h_fn(x):
        return h_np(x)
       
    def grad_fn(x):
        return grad_np(x)
    
    # Create obstacle constraint if inequality flag is set
    h_obs_fn = None
    grad_obs_fn = None
    obstacle_radius = 0.5 # Default radius for visualization
    obstacle_semi_axes = None  # For ellipsoid obstacle
    obstacle_spheres = None  # For multiple sphere obstacles
    obstacle_plane = None  # For halfspace obstacle
    obstacle_box = None  # For rectangular box obstacle
    if use_inequality:
        if constraint_name == 'sphere':
            # Cylinder obstacle along Z axis, traversing the sphere
            obstacle_radius = 0.75

            h_obs_fn, grad_obs_fn = make_cylinder_obstacle(radius=obstacle_radius, axis='z', center=[0, 0])
            print(f'\n--- Inequality constraint enabled: Cylinder obstacle (r={obstacle_radius}, axis=z) ---')
        elif constraint_name == 'ellipsoid':
            # Cylinder obstacle along Z axis, same as sphere
            obstacle_radius = 0.75

            h_obs_fn, grad_obs_fn = make_cylinder_obstacle(radius=obstacle_radius, axis='z', center=[0, 0])
            print(f'\n--- Inequality constraint enabled: Cylinder obstacle (r={obstacle_radius}, axis=z) ---')
        elif constraint_name == 'cylinder':
            # Rectangular box obstacle at center, traversing through cylinder
            h_obs_fn, grad_obs_fn = make_rectangular_box_obstacle(x_half=2, y_half=0.3, z_half=1, center=[0, 0, 0])
            obstacle_box = {'x_half': 2, 'y_half': 0.3, 'z_half': 1, 'center': [0, 0, 0]}
            print(f'\n--- Inequality constraint enabled: Rectangular box obstacle (center, size: 4.0x0.6x1.0) ---')
        else:
            print(f'\n--- Warning: Inequality constraint not implemented for {constraint_name} ---')   
      
    
    # ========================================================================
    # SINUSOIDAL MODE: Reference Tracking Experiment
    # ========================================================================
    if sinusoidal_mode:
          # Define learned wrappers for h and grad
        base_path = r"D:/arbol/Documents/LASAlab/TFG-1/TFG-LASA/project/experiments/3D"
        model_path = os.path.join(
            base_path, constraint_name,
            'outputs', 'regularizer_experiments', 'eikonal', 'noise_0.03_NN_trajectory_35', 'models',
            f'{constraint_name}_NN_eikonal_noise_0.03.pth'
        )
        print(f'\n--- Expected learned model path for QP: {model_path} ---')
        if not os.path.exists(model_path):
            print(f"\n--- ERROR: learned model not found at expected path: {model_path} ---")
            print("--- To use QP with learned constraints, first train a model using the 3D training pipeline (e.g. main_3d.py) and ensure the .pth is saved to the experiment's models folder.")
            return None
        try:
            model = load_learned_surface(model_path)
            if model is None:
                raise RuntimeError("load_learned_surface returned None")
            model.eval()
        except Exception as e:
            print(f"\n--- ERROR: failed to load learned model for QP: {e} ---")
            return None

        from src.data.datasetcontroller import h_learned, grad_learned

        def h_fn_learned(x):
            return h_learned(model, x.reshape(1, -1))[0]

        def grad_fn_learned(x):
            return grad_learned(model, x.reshape(1, -1))[0]
        # Run sinusoidal experiment using modular implementation
        if constraint_name != 'sphere':
            print(f'\n--- WARNING: Sinusoidal mode only implemented for sphere, got {constraint_name} ---')
            return None
        
        return run_sinusoidal_experiment(
            h_fn=h_fn,
            grad_fn=grad_fn,
            model=model,
            h_fn_learned=h_fn_learned,
            grad_fn_learned=grad_fn_learned,
            output_dir=output_dir,
            R=1.0,
            N_ref=100,
            amplitude=None,  
            freq=3,
            seed=seed,
            show_plots=show_plots
        )
    
    # ========================================================================
    # NORMAL MODE: Continue with standard experiment
    # ========================================================================
    
    # Initialize controller
    controller = ImitationController(input_size=6, hidden_size=32, output_size=3)
    controller_path = os.path.join(output_dir, 'controller.pth')
    
    if qp_mode:
        # QP mode: load existing controller
        if not os.path.exists(controller_path):
            print(f'\n--- ERROR: Controller not found at {controller_path} ---')
            print('--- Please train the controller first without --qp flag ---')
            return None
        
        print(f'\n--- Loading trained controller from: {controller_path} ---')
        controller.load_state_dict(torch.load(controller_path))
        controller.eval()
        print('--- Controller loaded successfully! ---')
        
        # Load model for QP evaluation (learned-only behavior)
        base_path = r"D:/arbol/Documents/LASAlab/TFG-1/TFG-LASA/project/experiments/3D"
        model_path = os.path.join(
            base_path, constraint_name,
            'outputs', 'regularizer_experiments', 'eikonal', 'noise_0.03_NN_trajectory_35', 'models',
            f'{constraint_name}_NN_eikonal_noise_0.03.pth'
        )
        print(f'\n--- Expected learned model path for QP: {model_path} ---')
        if not os.path.exists(model_path):
            print(f"\n--- ERROR: learned model not found at expected path: {model_path} ---")
            print("--- To use QP with learned constraints, first train a model using the 3D training pipeline (e.g. main_3d.py) and ensure the .pth is saved to the experiment's models folder.")
            return None
        try:
            model = load_learned_surface(model_path)
            if model is None:
                raise RuntimeError("load_learned_surface returned None")
            model.eval()
        except Exception as e:
            print(f"\n--- ERROR: failed to load learned model for QP: {e} ---")
            return None

        # Define learned wrappers for h and grad
        from src.data.datasetcontroller import h_learned, grad_learned

        def h_fn_learned(x):
            return h_learned(model, x.reshape(1, -1))[0]

        def grad_fn_learned(x):
            return grad_learned(model, x.reshape(1, -1))[0]
        
    else:
        # Non-QP flow: either train the controller (if cont=True) or load an existing one
        if cont:
            print('\n--- Training controller ---')
            # Show samples
            print('\n--- First 3 training samples ---')
            for i in range(min(3, len(states_train))):
                print(f'Sample {i}: Current {states_train[i, :3]}, Goal {states_train[i, 3:]}, Next {targets_train[i]}')

            gt_surface_dir = os.path.join(output_dir,'surface_trajectory_gt.png')
            plot_gt_surface_with_trajectories(h_fn, trajectories_train, save_path=gt_surface_dir, show=show_plots,
                                          bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48)
            # Train controller
            train_losses, test_losses, eval_epochs = train_controller(
                controller, states_train, targets_train, 
                states_test=states_test, targets_test=targets_test,
                num_epochs=num_epochs, batch_size=32, lr=0.001
            )

            # Plot learning curves
            results = {
                ('ImitationController', 'MSE'): (controller, {
                    'train_losses': np.array(train_losses),
                    'test_losses': np.array(test_losses),
                    'eval_epochs': np.array(eval_epochs)
                })
            }
            loss_plot_path = os.path.join(output_dir, 'learning_curves.png')
            plot_learning_curves(results, save_path=loss_plot_path)

            # Save controller
            torch.save(controller.state_dict(), controller_path)
            print(f'\n--- Controller saved to: {controller_path} ---')
        else:
            # Attempt to load a pre-trained controller for evaluation
            if os.path.exists(controller_path):
                print(f'\n--- Loading trained controller from: {controller_path} ---')
                try:
                    controller.load_state_dict(torch.load(controller_path))
                    controller.eval()
                    print('--- Controller loaded successfully! ---')
                except Exception as e:
                    print(f'--- ERROR: failed to load controller: {e} ---')
                    print('--- Run with --cont to train a new controller or ensure controller.pth exists in output folder ---')
                    return None
            else:
                print(f'--- No trained controller found at {controller_path} ---')
                print('--- Run this script with --cont to train the controller first. Example:')
                print('    python cont.py --constraint <constraint> --cont')
                return None
    
    # Create wrapper functions for h and grad_h if using QP
    if qp_mode:
        from src.data.datasetcontroller import h_learned, grad_learned
        
        def h_fn_learned(x):
            """Wrapper for h(x) evaluation."""
            return h_learned(model, x.reshape(1, -1))[0]
        
        def grad_fn_learned(x):
            """Wrapper for grad_h(x) evaluation."""
            return grad_learned(model, x.reshape(1, -1))[0]

    # If MPC is requested, try to locate a learned surface model and create learned h/grad wrappers
    # Don't clobber learned wrappers created above (QP mode)
    if 'h_fn_learned' not in locals():
        learned_model_available = False
        grad_fn_learned = None
        h_fn_learned = None
    if use_mpc:
        # Construct expected model path used for QP evaluations
        model_path = os.path.join(
            os.path.dirname(__file__), constraint_name,
            'outputs', 'regularizer_experiments', 'eikonal', 'noise_0.03_NN_trajectory_35', 'models',
            f'{constraint_name}_NN_eikonal_noise_0.03.pth'
        )
        if os.path.exists(model_path):
            try:
                print(f'\n--- Loading learned surface for MPC from: {model_path} ---')
                model = load_learned_surface(model_path)
                model.eval()
                from src.data.datasetcontroller import h_learned, grad_learned

                def h_fn_learned(x):
                    return h_learned(model, x.reshape(1, -1))[0]

                def grad_fn_learned(x):
                    return grad_learned(model, x.reshape(1, -1))[0]

                learned_model_available = True
                print('--- Learned surface loaded: MPC will use learned gradient for linearized constraints ---')
            except Exception as e:
                print(f'--- Warning: failed to load learned surface for MPC: {e} ---')
        else:
            print('\n--- No learned surface model found for MPC; will use GT gradient if available ---')
    
    # Evaluate on test trajectories
    print(f'\n--- Evaluating on {len(trajectories_test)} test trajectories ---')
    
    # If MPC-only mode, skip NN/QP trajectory generation
    if use_mpc and not qp_mode:
        print('\n--- MPC-only mode: skipping NN trajectory generation ---')
        nn_trajectories = None
        qp_trajectories = None
        real_traj_arrays = [np.asarray(traj) for traj in trajectories_test]
    elif qp_mode:
        # For QP mode produce both the raw NN rollout and the projected rollout
        nn_trajectories = []
        qp_trajectories = []
        real_traj_arrays = []
        for idx, real_traj in enumerate(trajectories_test):
            real_traj = np.asarray(real_traj)
            real_traj_arrays.append(real_traj)
            start_pt = real_traj[0].astype(float)
            goal_pt = real_traj[-1].astype(float)

            # Unconstrained NN rollout
            nn_traj = generate_trajectory(controller, start_pt, goal_pt, max_steps=len(real_traj))

            # Projected rollout using QP/Newton (or other method inside function)
            qp_traj = generate_trajectory_qp(
                controller, start_pt, goal_pt,
                h_fn_learned, grad_fn_learned,
                max_steps=len(real_traj)
            )

            nn_trajectories.append(np.asarray(nn_traj))
            qp_trajectories.append(np.asarray(qp_traj))
    else:
        # Non-QP mode: compute only NN (unprojected) rollouts
        nn_trajectories = []
        real_traj_arrays = []
        for traj_idx, real_traj in enumerate(trajectories_test):
            real_traj = np.asarray(real_traj)
            start_pt = real_traj[0].astype(float)
            goal_pt = real_traj[-1].astype(float)
            real_traj_arrays.append(real_traj)

            nn_traj = generate_trajectory(controller, start_pt, goal_pt, max_steps=len(real_traj))
            nn_trajectories.append(np.asarray(nn_traj))

    # If requested, compute MPC rollouts with synthetic test points that traverse obstacle
    mpc_trajectories = None
    mpc_trajectories_no_obs = None  # MPC without obstacle for comparison
    # container for MPC-specific metrics computed during the MPC block
    mpc_metrics = {}
    if use_mpc:
        print('\n--- Computing MPC rollouts (open-loop) as trajectory optimizer (start -> goal) ---')
        mpc_trajectories = []
        
        # If inequality constraints are active, also compute MPC without obstacle for comparison
        if use_inequality and h_obs_fn is not None:
            mpc_trajectories_no_obs = []
            print('--- Also computing MPC without obstacle for comparison ---')
        
        # Helper function to check if a line segment intersects the obstacle
        def line_intersects_obstacle(p1, p2, h_obs_fn, num_samples=20):
            """Check if line segment from p1 to p2 passes through obstacle interior"""
            for t in np.linspace(0, 1, num_samples):
                pt = p1 + t * (p2 - p1)
                h_val = float(h_obs_fn(pt))
                if h_val < 0:  # Inside obstacle (h < 0 means forbidden)
                    return True
            return False
        
        # Generate test point pairs for MPC trajectories
        # If obstacle is active: pairs that traverse obstacle
        # If no obstacle: use real trajectory endpoints
        test_pairs = []
        
        if h_obs_fn is not None:
            # Generate synthetic test points around the obstacle to force traversal
            if constraint_name in ['sphere', 'ellipsoid']:
                # For sphere/ellipsoid with cylinder obstacle along Z axis
                # Place pairs of points on opposite sides (front/back, left/right)
                angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions
                radius_surface = 1.0  # Sphere radius
                z_positions = [0.0, 0.3, -0.3]  # Different heights
                
                for angle_deg in angles:
                    for z_pos in z_positions:
                        angle_rad = np.deg2rad(angle_deg)
                        
                        # Point on one side of cylinder
                        x1 = radius_surface * np.cos(angle_rad)
                        y1 = radius_surface * np.sin(angle_rad)
                        z1 = z_pos
                        p1 = np.array([x1, y1, z1])
                        
                        # Point on opposite side
                        x2 = radius_surface * np.cos(angle_rad + np.pi)
                        y2 = radius_surface * np.sin(angle_rad + np.pi)
                        z2 = z_pos
                        p2 = np.array([x2, y2, z2])
                        
                        # Project onto surface
                        for _ in range(50):
                            h_val = float(h_fn(p1))
                            if abs(h_val) < 1e-6:
                                break
                            grad = np.asarray(grad_fn(p1)).reshape(-1)
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-8:
                                p1 = p1 - (h_val / (grad_norm**2)) * grad
                        
                        for _ in range(50):
                            h_val = float(h_fn(p2))
                            if abs(h_val) < 1e-6:
                                break
                            grad = np.asarray(grad_fn(p2)).reshape(-1)
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-8:
                                p2 = p2 - (h_val / (grad_norm**2)) * grad
                        
                        # Check if both points are outside obstacle and line traverses it
                        h_start = float(h_obs_fn(p1))
                        h_goal = float(h_obs_fn(p2))
                        
                        if h_start >= 0 and h_goal >= 0 and line_intersects_obstacle(p1, p2, h_obs_fn):
                            test_pairs.append((p1.copy(), p2.copy()))
            
            elif constraint_name == 'cylinder':
                # For cylinder with rectangular box obstacle
                # Place pairs at different Z heights on opposite sides
                z_positions = [-0.3, 0.0, 0.3, 0.6, -0.6]
                angles = [0, 45, 90, 135]  # 4 directions (will create 8 with opposites)
                radius = 0.5  # Cylinder radius
                
                for z_pos in z_positions:
                    for angle_deg in angles:
                        angle_rad = np.deg2rad(angle_deg)
                        
                        # Point on one side
                        x1 = radius * np.cos(angle_rad)
                        y1 = radius * np.sin(angle_rad)
                        z1 = z_pos
                        p1 = np.array([x1, y1, z1])
                        
                        # Point on opposite side
                        x2 = radius * np.cos(angle_rad + np.pi)
                        y2 = radius * np.sin(angle_rad + np.pi)
                        z2 = z_pos
                        p2 = np.array([x2, y2, z2])
                        
                        # Project onto cylinder surface
                        for _ in range(50):
                            h_val = float(h_fn(p1))
                            if abs(h_val) < 1e-6:
                                break
                            grad = np.asarray(grad_fn(p1)).reshape(-1)
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-8:
                                p1 = p1 - (h_val / (grad_norm**2)) * grad
                        
                        for _ in range(50):
                            h_val = float(h_fn(p2))
                            if abs(h_val) < 1e-6:
                                break
                            grad = np.asarray(grad_fn(p2)).reshape(-1)
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-8:
                                p2 = p2 - (h_val / (grad_norm**2)) * grad
                        
                        # Check if both points are outside obstacle and line traverses it
                        h_start = float(h_obs_fn(p1))
                        h_goal = float(h_obs_fn(p2))
                        
                        if h_start >= 0 and h_goal >= 0 and line_intersects_obstacle(p1, p2, h_obs_fn):
                            test_pairs.append((p1.copy(), p2.copy()))
        else:
            # No obstacle: use real trajectory endpoints
            for idx, real_traj in enumerate(real_traj_arrays):
                real_traj = np.asarray(real_traj).astype(float)
                x0 = real_traj[0]
                x_goal = real_traj[-1]
                test_pairs.append((x0.copy(), x_goal.copy()))
                print(f'  Added real trajectory {idx+1} for MPC test ✓')
        
        # If obstacle exists, also filter real trajectories that traverse it
        if h_obs_fn is not None:
            for idx, real_traj in enumerate(real_traj_arrays):
                real_traj = np.asarray(real_traj).astype(float)
                x0 = real_traj[0]
                x_goal = real_traj[-1]
                
                h_start = float(h_obs_fn(x0))
                h_goal = float(h_obs_fn(x_goal))
                
                if h_start >= 0 and h_goal >= 0 and line_intersects_obstacle(x0, x_goal, h_obs_fn):
                    test_pairs.append((x0.copy(), x_goal.copy()))
                    print(f'  Added real trajectory {idx+1}: traverses obstacle ✓')
        
        print(f'\n--- Generated {len(test_pairs)} test trajectory pairs that traverse obstacle ---')
        
        # Limit to 3 trajectories for cleaner visualization
        test_pairs = test_pairs[:3]
        print(f'--- Using {len(test_pairs)} trajectories for MPC tests ---')
        
        # Store test points for visualization
        mpc_test_points = []  # Will store (start, goal) tuples
        
        # Solve MPC for each test pair
        for idx, (x0, x_goal) in enumerate(test_pairs):
            mpc_test_points.append((x0.copy(), x_goal.copy()))
            # Estimate horizon based on distance
            dist = np.linalg.norm(x_goal - x0)
            N = max(15, min(40, int(dist * 20)))  # Adaptive horizon
            
            cfg = MPCConfig(N=N, dt=1.0)
            mpc = MPCController(cfg)

            # MPC as trajectory optimizer
            grad_to_use = grad_fn_learned 
            grad_label = 'learned'
            h_to_use = h_fn_learned 
            
            # MPC WITH obstacle (if --ie is active)
            u_seq, x_seq, info = mpc.solve(x0, x_goal=x_goal, grad_h_fn=grad_to_use, h_fn=h_to_use,
                                           grad_obs_fn=grad_obs_fn, h_obs_fn=h_obs_fn)
            print(f'  MPC open-loop (opt, grad={grad_label}): horizon={N}, status={info.get("status")}, time={info.get("solve_time")}')
            mpc_trajectories.append(np.asarray(x_seq))
            
            # MPC WITHOUT obstacle for comparison (if --ie is active)
            if use_inequality and h_obs_fn is not None:
                u_seq_no_obs, x_seq_no_obs, info_no_obs = mpc.solve(x0, x_goal=x_goal, grad_h_fn=grad_to_use, h_fn=h_to_use,
                                                                      grad_obs_fn=None, h_obs_fn=None)
                print(f'  MPC no-obs (opt, grad={grad_label}): horizon={N}, status={info_no_obs.get("status")}, time={info_no_obs.get("solve_time")}')
                mpc_trajectories_no_obs.append(np.asarray(x_seq_no_obs))
        
        print(f'--- MPC rollouts computed: {len(mpc_trajectories)} trajectories ---')
        if mpc_trajectories_no_obs:
            print(f'--- MPC no-obstacle rollouts computed: {len(mpc_trajectories_no_obs)} trajectories ---')

        # Compute distances of MPC trajectories to the REAL constraint surface (gt_samples)
        import csv
        
        # Function to compute distance summary for a set of trajectories
        def compute_distance_summary(trajectories):
            dist_summary_rows = []
            
            for ti, x_seq in enumerate(trajectories):
                # x_seq shape (N+1, dim_x)
                hs = []
                grads_norm = []
                approx_dists = []
                euclid_dists = []

                for k, x in enumerate(x_seq):
                    # h and grad (GT)
                    try:
                        hval = float(h_np(x))
                    except Exception:
                        hval = float(h_fn(x)) if 'h_fn' in locals() else 0.0
                    try:
                        gradv = np.asarray(grad_np(x)).reshape(-1)
                        gnorm = float(np.linalg.norm(gradv))
                    except Exception:
                        gnorm = 0.0

                    # first-order approximate distance
                    if gnorm > 1e-12:
                        approx_dist = abs(hval) / gnorm
                    else:
                        approx_dist = abs(hval)

                    # exact Euclidean distance via KD-tree if available
                    if use_kdtree:
                        try:
                            d_e, _ = tree.query(np.asarray(x).reshape(1, -1))
                            euclid_dist = float(d_e[0])
                        except Exception:
                            euclid_dist = float('nan')
                    else:
                        euclid_dist = float('nan')

                    hs.append(hval)
                    grads_norm.append(gnorm)
                    approx_dists.append(approx_dist)
                    euclid_dists.append(euclid_dist)

                # Prefer Euclidean mean distance if available, otherwise fallback to approx
                if all(not np.isnan(d) for d in euclid_dists):
                    mean_dist = float(np.mean(euclid_dists))
                    max_dist = float(np.max(euclid_dists))
                else:
                    mean_dist = float(np.mean(approx_dists))
                    max_dist = float(np.max(approx_dists))

                # final_dist: distance of last point to surface
                final_dist = float(approx_dists[-1]) if len(approx_dists) > 0 else float('nan')

                dist_summary_rows.append((ti, mean_dist, max_dist, final_dist))
            
            return dist_summary_rows

        # Try to use a KD-tree on gt_samples for exact Euclidean distances
        use_kdtree = False
        try:
            from scipy.spatial import cKDTree
            if 'gt_samples' in locals() and gt_samples is not None and len(gt_samples) > 0:
                tree = cKDTree(np.asarray(gt_samples))
                use_kdtree = True
        except Exception:
            use_kdtree = False

        # Compute summary for MPC with obstacle
        dist_summary_rows = compute_distance_summary(mpc_trajectories)

        # Save summary CSV for all MPC trajectories
        if use_inequality and mpc_trajectories_no_obs is not None:
            # Save two summaries: one for MPC with obstacle and one without
            summary_path_with_obs = os.path.join(output_dir, 'mpc_surface_distance_summary_with_obstacle.csv')
            with open(summary_path_with_obs, 'w', newline='') as sf:
                sw = csv.writer(sf)
                sw.writerow(['traj_index', 'mean_dist', 'max_dist', 'final_dist'])
                for row in dist_summary_rows:
                    sw.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
            print(f'--- MPC (with obstacle) surface distance summary saved to: {summary_path_with_obs} ---')
            
            # Compute summary for MPC without obstacle
            dist_summary_rows_no_obs = compute_distance_summary(mpc_trajectories_no_obs)
            summary_path_no_obs = os.path.join(output_dir, 'mpc_surface_distance_summary_no_obstacle.csv')
            with open(summary_path_no_obs, 'w', newline='') as sf:
                sw = csv.writer(sf)
                sw.writerow(['traj_index', 'mean_dist', 'max_dist', 'final_dist'])
                for row in dist_summary_rows_no_obs:
                    sw.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
            print(f'--- MPC (no obstacle) surface distance summary saved to: {summary_path_no_obs} ---')
            
            # Compute metrics for both
            try:
                all_means_with = [r[1] for r in dist_summary_rows]
                all_means_no = [r[1] for r in dist_summary_rows_no_obs]
                if len(all_means_with) > 0:
                    mpc_metrics['mean_surface_distance_with_obstacle'] = float(np.mean(all_means_with))
                    print(f"--- MPC (with obstacle) mean surface distance: {mpc_metrics['mean_surface_distance_with_obstacle']:.6f} ---")
                if len(all_means_no) > 0:
                    mpc_metrics['mean_surface_distance_no_obstacle'] = float(np.mean(all_means_no))
                    print(f"--- MPC (no obstacle) mean surface distance: {mpc_metrics['mean_surface_distance_no_obstacle']:.6f} ---")
            except Exception as e:
                print(f'--- Warning: could not compute MPC summary metrics: {e} ---')
        else:
            # Normal case: single summary
            summary_path = os.path.join(output_dir, 'mpc_surface_distance_summary.csv')
            with open(summary_path, 'w', newline='') as sf:
                sw = csv.writer(sf)
                sw.writerow(['traj_index', 'mean_dist', 'max_dist', 'final_dist'])
                for row in dist_summary_rows:
                    sw.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])
            print(f'--- MPC surface distance summaries saved to: {summary_path} ---')

            # Compute and store a single summary metric: mean of per-trajectory mean distances
            try:
                all_means = [r[1] for r in dist_summary_rows]
                if len(all_means) > 0:
                    mpc_metrics['mean_surface_distance'] = float(np.mean(all_means))
                    print(f"--- MPC mean surface distance (over trajectories): {mpc_metrics['mean_surface_distance']:.6f} ---")
            except Exception as e:
                print(f'--- Warning: could not compute MPC summary metrics: {e} ---')


    # Generate test trajectories using GT constraint and plot comparison
    # Always show: Real and NN. Add QP and MPC curves only if flags were used.
    comparison_2D_plot_path = os.path.join(output_dir, 'trajectory_comaprison_2D.png')
    comparison_plot_path = os.path.join(output_dir, 'trajectory_comparison.png')
    comparison_learned_plot_path = os.path.join(output_dir, 'trajectory_comparison_learned.png')

    # Always generate 2D projection plot
    # For MPC with obstacle, don't show real trajectories (use empty list)
    if use_mpc and use_inequality:
        real_traj_for_2d = []  # Don't show real trajectories
        nn_traj_for_2d = []    # Don't show NN either
        mpc_no_obs_for_comparison = mpc_trajectories_no_obs if 'mpc_trajectories_no_obs' in locals() else None
    else:
        real_traj_for_2d = real_traj_arrays
        nn_traj_for_2d = nn_trajectories
        mpc_no_obs_for_comparison = None
    
    plot_projection_comparison(
        real_traj_for_2d,
        nn_traj_for_2d,
        constraint_name,
        qp_trajectories=qp_trajectories if qp_mode else None,
        mpc_trajectories=mpc_trajectories if use_mpc else None,
        mpc_trajectories_no_obs=mpc_no_obs_for_comparison,
        save_path=comparison_2D_plot_path,
        show=show_plots,
        h_fn=h_fn
    )

    # 3D comparison: build learned dict with only the requested methods
    learned_methods = {}
    if nn_trajectories is not None:
        learned_methods['Imitation'] = nn_trajectories
    if qp_mode and qp_trajectories is not None:
        learned_methods['QP'] = qp_trajectories
    if use_mpc and mpc_trajectories is not None:
        # If we have both MPC with and without obstacle, add both for comparison
        if use_inequality and mpc_trajectories_no_obs is not None:
            learned_methods['MPC (no obstacle)'] = mpc_trajectories_no_obs
            learned_methods['MPC (with obstacle)'] = mpc_trajectories
        else:
            learned_methods['MPC'] = mpc_trajectories

    # Plot with GT surface (skip if MPC-only and no NN)
    if len(learned_methods) > 0:
        # For MPC with obstacle avoidance, create synthetic "real" trajectories showing test points
        if use_mpc and use_inequality and 'mpc_test_points' in locals():
            # Create minimal trajectories (just start and goal) to show markers
            synthetic_real_trajs = []
            for start_pt, goal_pt in mpc_test_points[:min(len(mpc_test_points), len(mpc_trajectories))]:
                # Create a 2-point "trajectory" just to show start/goal markers
                synthetic_real_trajs.append(np.array([start_pt, goal_pt]))
            real_traj_to_plot = synthetic_real_trajs
        else:
            real_traj_to_plot = real_traj_arrays
        
        if len(learned_methods) == 1 and 'Imitation' in learned_methods:
            plot_trajectory_comparison(
                real_traj_to_plot, nn_trajectories, constraint_name,
                save_path=comparison_plot_path, show=show_plots,
                h_fn=h_fn, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                h_obs_fn=h_obs_fn, obstacle_radius=obstacle_radius, obstacle_semi_axes=obstacle_semi_axes,
                obstacle_spheres=obstacle_spheres, obstacle_plane=obstacle_plane, obstacle_box=obstacle_box
            )
        else:
            # If animated MPC is requested and we have MPC trajectories, generate animation
            # Check if any MPC method exists in learned_methods
            has_mpc = any('MPC' in method for method in learned_methods.keys()) if isinstance(learned_methods, dict) else False
            
            if animated_mpc and use_mpc and has_mpc:
                print('\n--- Generating animated MPC trajectory plot ---')
                # Generate both GIF (for quick preview) and MP4 (for PowerPoint)
                animation_path_gif = os.path.join(output_dir, 'trajectory_comparison_MPC_animated.gif')
                animation_path_mp4 = os.path.join(output_dir, 'trajectory_comparison_MPC_animated.mp4')
                
                # Determine which trajectories to use for animation
                # If we have obstacle, use MPC without obstacle as reference and MPC with obstacle as main
                # Otherwise, use real trajectories as reference
                if use_inequality and 'MPC (no obstacle)' in learned_methods and 'MPC (with obstacle)' in learned_methods:
                    # Use MPC without obstacle as reference trajectory
                    animation_real_traj = learned_methods['MPC (no obstacle)']
                    animation_methods = {'MPC (with obstacle)': learned_methods['MPC (with obstacle)']}
                    reference_label = 'MPC (no obstacle)'
                    print('Animating: MPC without obstacle (reference) vs MPC with obstacle')
                else:
                    # Use real trajectories as reference
                    animation_real_traj = real_traj_to_plot
                    reference_label = 'Original'
                    # Find the MPC method to use for animation
                    mpc_method_key = 'MPC (with obstacle)' if 'MPC (with obstacle)' in learned_methods else \
                                     'MPC (no obstacle)' if 'MPC (no obstacle)' in learned_methods else \
                                     next((k for k in learned_methods.keys() if 'MPC' in k), None)
                    if mpc_method_key:
                        animation_methods = {mpc_method_key: learned_methods[mpc_method_key]}
                
                if animation_methods:
                    # Check if ffmpeg is available, otherwise use GIF directly
                    import matplotlib.animation as manimation
                    has_ffmpeg = 'ffmpeg' in manimation.writers.list()
                    
                    if has_ffmpeg:
                        # Try MP4 (better quality for PowerPoint)
                        print('Using ffmpeg for MP4 export...')
                        plot_trajectory_comparison_MPC_animated(
                            animation_real_traj, animation_methods, constraint_name,
                            save_path=animation_path_mp4,
                            h_fn=h_fn, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                            h_obs_fn=h_obs_fn, obstacle_radius=obstacle_radius, obstacle_semi_axes=obstacle_semi_axes,
                            obstacle_spheres=obstacle_spheres, obstacle_plane=obstacle_plane, obstacle_box=obstacle_box,
                            interval=100,
                            reference_label=reference_label
                        )
                        print(f'✓ Animated MPC plot (MP4) saved to: {animation_path_mp4}')
                        print(f'  To use in PowerPoint: Insert → Video → This Device → Select the .mp4 file')
                    else:
                        # Use GIF (Pillow writer)
                        print('ffmpeg not available, using GIF format (Pillow writer)...')
                        plot_trajectory_comparison_MPC_animated(
                            animation_real_traj, animation_methods, constraint_name,
                            save_path=animation_path_gif,
                            h_fn=h_fn, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                            h_obs_fn=h_obs_fn, obstacle_radius=obstacle_radius, obstacle_semi_axes=obstacle_semi_axes,
                            obstacle_spheres=obstacle_spheres, obstacle_plane=obstacle_plane, obstacle_box=obstacle_box,
                            interval=100,
                            reference_label=reference_label
                        )
                        print(f'✓ Animated MPC plot (GIF) saved to: {animation_path_gif}')
                        print(f'  To use in PowerPoint: Insert → Pictures → This Device → Select the .gif file')
                        print(f'--- Animated MPC plot (GIF) saved to: {animation_path_gif} ---')
                        print(f'--- To use in PowerPoint: Insert → Pictures → This Device → Select the .gif file ---')
            else:
                # Normal static plot
                plot_trajectory_comparison(
                    real_traj_to_plot, learned_methods, constraint_name,
                    save_path=comparison_plot_path, show=show_plots,
                    h_fn=h_fn, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                    h_obs_fn=h_obs_fn, obstacle_radius=obstacle_radius, obstacle_semi_axes=obstacle_semi_axes,
                    obstacle_spheres=obstacle_spheres, obstacle_plane=obstacle_plane, obstacle_box=obstacle_box
                )
    
    # Plot with LEARNED surface (if MPC or QP used, we have the learned model)
    if use_mpc or qp_mode:
        # Check if we have h_fn_learned available
        if 'h_fn_learned' in locals() and h_fn_learned is not None and len(learned_methods) > 0:
            print('\n--- Generating trajectory comparison plot with LEARNED surface ---')
            # Always use dict format for consistency
            """
            plot_trajectory_comparison(
                real_traj_arrays, learned_methods, constraint_name,
                save_path=comparison_learned_plot_path, show=show_plots,
                h_fn=h_fn_learned, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                h_obs_fn=h_obs_fn, obstacle_radius=obstacle_radius, obstacle_semi_axes=obstacle_semi_axes,
                obstacle_spheres=obstacle_spheres, obstacle_plane=obstacle_plane
            )
            print(f'--- Learned surface plot saved to: {comparison_learned_plot_path} ---')
    """
    # Evaluate all test trajectories and produce summaries per method (NN, QP, MPC)
    summaries = {}

    # Ensure we have the learned_methods dict constructed earlier
    # learned_methods: dict mapping method name -> list of trajectories
    for method, trajs in learned_methods.items():
        # Skip test_trajectory_summary for MPC (only generate surface distance summary)
        if 'MPC' in method:
            continue
            
        if method == 'QP':
            # For QP we evaluate summaries and include GT surface distance (h_fn/grad_fn)
            try:
                summary_m = evaluate_all_test_trajectories(
                    real_traj_arrays, trajs, qp=True,
                    h_surface_fn=h_fn, grad_surface_fn=grad_fn
                )
            except Exception:
                summary_m = evaluate_all_test_trajectories(
                    real_traj_arrays, trajs, qp=True,
                    h_surface_fn=h_fn, grad_surface_fn=grad_fn
                )
        else:
            # For NN and other methods include GT surface distance metric
            summary_m = evaluate_all_test_trajectories(real_traj_arrays, trajs,
                                                      h_surface_fn=h_fn, grad_surface_fn=grad_fn)

        # If we computed MPC-specific metrics earlier, inject them into the summary so they
        # are written to per-method CSVs and included in consolidated summaries.
        if method == 'MPC' and 'mean_surface_distance' in mpc_metrics:
            try:
                summary_m['mean_surface_distance'] = mpc_metrics['mean_surface_distance']
            except Exception:
                pass

        summaries[method] = summary_m

        # Save per-method CSV (skip for MPC as it has its own surface distance summary)
        import csv
        csv_name = f'test_trajectory_summary_{method.lower()}.csv'
        csv_path = os.path.join(output_dir, csv_name)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            keys = ['avg_mean_distance', 'avg_max_distance', 'avg_final_error', 'avg_path_length_ratio', 'best_mean_distance', 'worst_mean_distance']
            # append MPC-specific mean-to-GT-surface metric if present
            if 'mean_surface_distance' in summary_m:
                keys.append('mean_surface_distance')

            for key in keys:
                val = summary_m.get(key, float('nan'))
                # format floats with 3 decimals, keep nan as string
                if isinstance(val, float) and np.isnan(val):
                    vstr = 'nan'
                else:
                    try:
                        vstr = f"{float(val):.3f}"
                    except Exception:
                        vstr = str(val)
                writer.writerow([key, vstr])
        print(f'\n--- Summary metrics saved to: {csv_path} ---')

    # Return summaries: single dict if only one method, else dict of summaries
    if len(summaries) == 1:
        return list(summaries.values())[0]
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/evaluate controller from saved trajectories')
    parser.add_argument('--constraint', type=str, default="sphere", 
                        choices=['sphere', 'ellipsoid', 'cylinder', 'torus', 'all'],
                        help='Constraint name')
    parser.add_argument('--trajectories_path', type=str, default=None,
                        help='Path to saved trajectories pickle file')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs (default: 150)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    parser.add_argument('--qp', action='store_true',
                        help='Use QP-based projection (no training, only evaluation)')
    parser.add_argument('--mpc', action='store_true',
                        help='Use MPC-based projection/evaluation (no training)')
    parser.add_argument('--cont', action='store_true',
                        help='Train controller (pass this flag to train). If omitted, script will try to load existing controller.pth')
    parser.add_argument('--ie', action='store_true',
                        help='Enable inequality constraint (obstacle avoidance)')
    parser.add_argument('--sinusoidal', action='store_true',
                        help='Run sinusoidal reference tracking experiment (sphere only)')
    parser.add_argument('--animated', action='store_true',
                        help='Generate animated MPC trajectories (requires --mpc flag)')
    args = parser.parse_args()
    
    show_plots = not args.no_show
    
    # Determine constraints
    if args.constraint == 'all':
        constraints = ['sphere', 'ellipsoid', 'cylinder', 'torus']
    else:
        constraints = [args.constraint]
    
    # Run for each constraint
    all_summaries = {}
    
    for constraint in constraints:
        print(f'\n{"="*60}')
        print(f'Processing constraint: {constraint.upper()}')
        print(f'{"="*60}\n')
        
        summary = main(
            constraint_name=constraint,
            trajectories_path=args.trajectories_path,
            num_epochs=args.epochs,
            seed=args.seed,
            show_plots=show_plots,
            qp_mode=args.qp,
            use_mpc=args.mpc,
            cont=args.cont,
            use_inequality=args.ie,
            sinusoidal_mode=args.sinusoidal,
            animated_mpc=args.animated
        )
        
        if summary is not None:
            all_summaries[constraint] = summary
        
        print(f'\n{"="*60}')
        print(f'Completed: {constraint.upper()}')
        print(f'{"="*60}\n')
    
    # Consolidated summary if multiple constraints
    if len(all_summaries) > 1:
        import csv

        metric_keys = ['avg_mean_distance', 'avg_max_distance', 'avg_final_error', 'avg_path_length_ratio']

        def _extract_metric(summary_obj, key):
            # summary_obj may be a flat summary dict or a dict of method->summary
            if isinstance(summary_obj, dict) and key in summary_obj:
                return summary_obj[key]
            if isinstance(summary_obj, dict):
                # try to average across nested method summaries
                vals = []
                for v in summary_obj.values():
                    if isinstance(v, dict) and key in v:
                        vals.append(v[key])
                if len(vals) > 0:
                    return np.mean(vals)
            # fallback: signal missing metric by raising
            raise KeyError(key)

        def _get_method_value(summary_obj, method, key):
            """Return the metric value for a given method from a possibly nested summary.
            If the exact method summary exists, use it. Otherwise fall back to _extract_metric.
            If still missing, return nan.
            """
            # If summary_obj is nested method->summary and contains the method, use that
            if isinstance(summary_obj, dict) and method in summary_obj and isinstance(summary_obj[method], dict):
                return summary_obj[method].get(key, float('nan'))
            # Otherwise try extracting a representative metric (average across methods or flat)
            try:
                return _extract_metric(summary_obj, key)
            except KeyError:
                return float('nan')

        # Determine which consolidated files to produce
        targets = []  # list of tuples (label, filename, method)
        base_dir = os.path.dirname(__file__)
        if args.qp:
            targets.append(('QP', os.path.join(base_dir, 'consolidated_summary_qp.csv'), 'QP'))
        if args.mpc:
            targets.append(('MPC', os.path.join(base_dir, 'consolidated_summary_mpc.csv'), 'MPC'))
        # If neither qp nor mpc flags were provided, create a generic consolidated_summary.csv
        if not targets:
            targets.append(('ALL', os.path.join(base_dir, 'consolidated_summary.csv'), None))

        for label, summary_path, method in targets:
            # Compute averages per metric across constraints
            # allow adding MPC-specific metric to consolidated output
            local_metric_keys = list(metric_keys)
            # If any per-constraint summary contains the mean_surface_distance key,
            # include it in the consolidated output for non-MPC targets as well.
            try:
                has_msd = False
                for s in all_summaries.values():
                    if isinstance(s, dict) and 'mean_surface_distance' in s:
                        has_msd = True
                        break
                    if isinstance(s, dict):
                        for v in s.values():
                            if isinstance(v, dict) and 'mean_surface_distance' in v:
                                has_msd = True
                                break
                    if has_msd:
                        break
                if has_msd and 'mean_surface_distance' not in local_metric_keys:
                    local_metric_keys.append('mean_surface_distance')
            except Exception:
                pass
            if method == 'MPC':
                # MPC compares trajectories to the surface, not to other trajectories.
                # Remove trajectory-to-trajectory metrics: avg_mean_distance, avg_max_distance, avg_path_length_ratio
                for rem in ('avg_mean_distance', 'avg_max_distance', 'avg_path_length_ratio','avg_final_error'):
                    if rem in local_metric_keys:
                        local_metric_keys.remove(rem)
                # Keep avg_final_error as a possible numeric diagnostic, then add mean_surface_distance
                if 'mean_surface_distance' not in local_metric_keys:
                    local_metric_keys.append('mean_surface_distance')
                # Also include mean of final distance-to-goal for MPC
                if 'mean_final_dist' not in local_metric_keys:
                    local_metric_keys.append('mean_final_dist')

            # If MPC consolidation requested, attempt to read each constraint's
            # controller_outputs/mpc_surface_distance_summary.csv and compute the
            # mean of the per-trajectory mean distances. This is used as the
            # per-constraint 'mean_surface_distance' value.
            mpc_constraint_means = {}
            if method == 'MPC':
                for constraint in all_summaries.keys():
                    mpc_csv = os.path.join(base_dir, constraint, 'controller_outputs', 'mpc_surface_distance_summary.csv')
                    mean_val = float('nan')
                    try:
                        if os.path.exists(mpc_csv):
                            with open(mpc_csv, 'r', newline='') as mf:
                                rdr = csv.reader(mf)
                                hdr = next(rdr, None)
                                # expect header: traj_index, mean_dist, max_dist, final_dist
                                mean_vals = []
                                for r in rdr:
                                    try:
                                        mean_vals.append(float(r[1]))
                                    except Exception:
                                        continue
                                if len(mean_vals) > 0:
                                    mean_val = float(np.mean(mean_vals))
                    except Exception:
                        mean_val = float('nan')
                    mpc_constraint_means[constraint] = mean_val

            # Also compute mean of final_dist (distance final->goal) per constraint
            mpc_constraint_final_means = {}
            if method == 'MPC':
                for constraint in all_summaries.keys():
                    mpc_csv = os.path.join(base_dir, constraint, 'controller_outputs', 'mpc_surface_distance_summary.csv')
                    final_vals = []
                    try:
                        if os.path.exists(mpc_csv):
                            with open(mpc_csv, 'r', newline='') as mf:
                                rdr = csv.reader(mf)
                                hdr = next(rdr, None)
                                for r in rdr:
                                    try:
                                        # final_dist is expected at index 3
                                        final_vals.append(float(r[3]))
                                    except Exception:
                                        continue
                    except Exception:
                        pass
                    mpc_constraint_final_means[constraint] = float(np.mean(final_vals)) if len(final_vals) > 0 else float('nan')

            avg_metrics = {}
            # compute avg_metrics by aggregating per-constraint values (respecting MPC overrides)
            for key in local_metric_keys:
                vals = []
                for constraint, s in all_summaries.items():
                    if method is None:
                        try:
                            v = _extract_metric(s, key)
                        except KeyError:
                            continue
                    else:
                        # MPC special-case: use precomputed mpc_constraint_means for the 'mean_surface_distance' key
                        if method == 'MPC' and key == 'mean_surface_distance':
                            v = mpc_constraint_means.get(constraint, float('nan'))
                        elif method == 'MPC' and key == 'mean_final_dist':
                            v = mpc_constraint_final_means.get(constraint, float('nan'))
                        else:
                            v = _get_method_value(s, method, key)
                    if not (isinstance(v, float) and np.isnan(v)):
                        vals.append(v)
                avg_metrics[key] = np.mean(vals) if len(vals) > 0 else float('nan')

            # Write CSV for this target
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Constraint'] + local_metric_keys)
                for constraint, summary in all_summaries.items():
                    row_vals = []
                    for key in local_metric_keys:
                        if method is None:
                            try:
                                val = _extract_metric(summary, key)
                            except KeyError:
                                val = float('nan')
                        else:
                            if method == 'MPC' and key == 'mean_surface_distance':
                                val = mpc_constraint_means.get(constraint, float('nan'))
                            elif method == 'MPC' and key == 'mean_final_dist':
                                val = mpc_constraint_final_means.get(constraint, float('nan'))
                            else:
                                val = _get_method_value(summary, method, key)

                        # format numeric values; mark nan explicitly
                        if isinstance(val, float) and np.isnan(val):
                            row_vals.append('nan')
                        else:
                            try:
                                row_vals.append(f"{float(val):.3f}")
                            except Exception:
                                row_vals.append(str(val))
                    writer.writerow([constraint] + row_vals)
                writer.writerow(['AVERAGE'] + [f"{avg_metrics[key]:.3f}" if not (isinstance(avg_metrics[key], float) and np.isnan(avg_metrics[key])) else 'nan' for key in local_metric_keys])

            print(f'\nConsolidated summary ({label}): {summary_path}')
	