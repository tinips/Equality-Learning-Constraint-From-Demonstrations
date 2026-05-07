"""
Visualization utilities for controller training and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import os


def compute_isosurface(h_fn, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48):
    """Compute an isosurface mesh (verts, faces) for h_fn==0 using marching cubes.

    Returns (verts, faces) or (None, None) if marching cubes is unavailable or
    the function has no zero-crossing within the bounds.
    """
    try:
        from skimage.measure import marching_cubes
    except Exception:
        return None, None

    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[2], bounds[3], res)
    z = np.linspace(bounds[4], bounds[5], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    # Evaluate h_fn (vectorized if possible)
    try:
        vals = h_fn(pts)
    except Exception:
        vals = np.array([h_fn(p) for p in pts])

    try:
        GT = vals.reshape(X.shape)
    except Exception:
        return None, None

    if not (np.any(GT <= 0.0) and np.any(GT >= 0.0)):
        return None, None

    dx = (bounds[1] - bounds[0]) / (res - 1)
    dy = (bounds[3] - bounds[2]) / (res - 1)
    dz = (bounds[5] - bounds[4]) / (res - 1)

    try:
        verts, faces, _, _ = marching_cubes(GT, level=0.0, spacing=(dx, dy, dz))
        verts[:, 0] += bounds[0]
        verts[:, 1] += bounds[2]
        verts[:, 2] += bounds[4]
        return verts, faces
    except Exception:
        return None, None



def plot_trajectory_comparison(real_trajectories, learned_trajectories, constraint_name, save_path=None, show=True,
                               h_fn=None, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                               surface_alpha=0.4, surface_color='red', learned_h_fn=None, learned_surface_color='cyan',
                               h_obs_fn=None, obs_color='orange', obs_alpha=0.3, obstacle_radius=0.5, obstacle_semi_axes=None,
                               obstacle_spheres=None, obstacle_plane=None, obstacle_box=None, real_label='Real', learned_label=None):
    """Compare real vs learned trajectories in a single 3D plot.
    
    Args:
        real_trajectories: List of real trajectory arrays (max 3)
        learned_trajectories: List of learned trajectory arrays (max 3)
        constraint_name: Name of constraint
        save_path: Optional path to save figure
        show: Whether to display the plot (default: True)
        h_obs_fn: Optional obstacle function (h>=0 outside obstacle) to visualize forbidden zone
        obs_color: Color for obstacle visualization
        obs_alpha: Transparency for obstacle
        real_label: Custom label for real trajectories (default: 'Real')
        learned_label: Custom label for learned trajectories (default: None, uses method name)
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Support multiple learned-methods: learned_trajectories can be either
    # - a list of arrays (one learned set matching real_trajectories), or
    # - a dict mapping method name -> list of arrays (each list matching real_trajectories)
    multi_methods = isinstance(learned_trajectories, dict)
    methods = list(learned_trajectories.keys()) if multi_methods else ['Learned']

    # If we have MPC with and without obstacle, create a single combined plot
    # Otherwise, create one figure per learned method
    combine_mpc_comparison = multi_methods and 'MPC (no obstacle)' in methods and 'MPC (with obstacle)' in methods
    
    if combine_mpc_comparison:
        # Special case: show both MPC versions in the same plot
        methods_to_plot = ['MPC_combined']
        figs = []
    else:
        figs = []
    
    # Create one figure per learned method, each containing ALL real trajectories
    for method in (methods_to_plot if combine_mpc_comparison else methods):
        fig = plt.figure(figsize=(12, 9) if combine_mpc_comparison else (10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # display name mapping for legend/title (show 'projection' instead of 'QP')
        if combine_mpc_comparison:
            display_name = 'MPC Comparison'
        else:
            if method == 'QP':
                display_name = 'projection'
            elif method == 'Imitation':
                display_name = 'Imitation'
            else:
                display_name = method
    
        # Optionally plot GT isosurface if h_fn is provided
        if h_fn is not None:
            verts, faces = compute_isosurface(h_fn, bounds=bounds, res=res)
            if verts is not None and faces is not None:
                try:
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, alpha=surface_alpha, color='lightcoral')
                except Exception:
                    pass
        
        # If obstacle exists, draw the boundary circles where it intersects the surface
        if h_obs_fn is not None and h_fn is not None:
            # For cylinder obstacle - draw full cylinder visualization
            # This is much faster than filtering the full mesh
            theta = np.linspace(0, 2*np.pi, 100)
            
            # For sphere constraint with cylinder obstacle
            if constraint_name.lower() == 'sphere' and obstacle_semi_axes is None and obstacle_spheres is None:
                # Draw cylinder as vertical lines + top and bottom circles
                sphere_radius = 1.0  # Sphere constraint radius
                
                # Draw vertical lines showing cylinder boundary
                num_lines = 16  # Number of vertical lines to draw
                theta_lines = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
                z_range = np.linspace(-sphere_radius, sphere_radius, 50)
                
                for angle in theta_lines:
                    x_line = obstacle_radius * np.cos(angle)
                    y_line = obstacle_radius * np.sin(angle)
                    x_arr = np.full_like(z_range, x_line)
                    y_arr = np.full_like(z_range, y_line)
                    ax.plot(x_arr, y_arr, z_range, color=obs_color, linewidth=1.5, alpha=0.6)
                
                # Draw top and bottom circles (flat, not curved)
                theta_circle = np.linspace(0, 2*np.pi, 100)
                x_circle = obstacle_radius * np.cos(theta_circle)
                y_circle = obstacle_radius * np.sin(theta_circle)
                
                # Top circle at z = sphere_radius
                z_top = np.full_like(x_circle, sphere_radius)
                ax.plot(x_circle, y_circle, z_top, color=obs_color, linewidth=3.5, alpha=0.8)
                
                # Bottom circle at z = -sphere_radius
                z_bottom = np.full_like(x_circle, -sphere_radius)
                ax.plot(x_circle, y_circle, z_bottom, color=obs_color, linewidth=3.5, alpha=0.8)
            
            # For ellipsoid constraint with cylinder obstacle
            elif constraint_name.lower() == 'ellipsoid' and obstacle_semi_axes is None and obstacle_spheres is None:
                # Draw cylinder as vertical lines + top and bottom circles
                a_ellipsoid, b_ellipsoid, c_ellipsoid = 0.8, 1.2, 1.5
                
                # Draw vertical lines showing cylinder boundary
                num_lines = 16  # Number of vertical lines to draw
                theta_lines = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
                z_range = np.linspace(-c_ellipsoid, c_ellipsoid, 50)
                
                for angle in theta_lines:
                    x_line = obstacle_radius * np.cos(angle)
                    y_line = obstacle_radius * np.sin(angle)
                    x_arr = np.full_like(z_range, x_line)
                    y_arr = np.full_like(z_range, y_line)
                    ax.plot(x_arr, y_arr, z_range, color=obs_color, linewidth=2.5, alpha=0.6)
                
                # Draw top and bottom circles (flat, not curved)
                theta_circle = np.linspace(0, 2*np.pi, 100)
                x_circle = obstacle_radius * np.cos(theta_circle)
                y_circle = obstacle_radius * np.sin(theta_circle)
                
                # Top circle at z = c_ellipsoid
                z_top = np.full_like(x_circle, c_ellipsoid)
                ax.plot(x_circle, y_circle, z_top, color=obs_color, linewidth=3.5, alpha=0.8)
                
                # Bottom circle at z = -c_ellipsoid
                z_bottom = np.full_like(x_circle, -c_ellipsoid)
                ax.plot(x_circle, y_circle, z_bottom, color=obs_color, linewidth=3.5, alpha=0.8)
            
            
            # For ellipsoid constraint with two sphere obstacles
            elif constraint_name.lower() == 'ellipsoid' and obstacle_spheres is not None:
                # Draw circles at intersection of spheres with ellipsoid surface
                # The intersection forms a circle where both constraints are active
                a_ellipsoid, b_ellipsoid, c_ellipsoid = 0.8, 1.2, 1.5
                theta = np.linspace(0, 2*np.pi, 100)
                
                for cx, cy, cz, r_sphere in obstacle_spheres:
                    # Points on sphere: (x-cx)^2 + (y-cy)^2 + (z-cz)^2 = r^2
                    # Points on ellipsoid: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
                    # Intersection: circular ring where both are satisfied
                    
                    # Generate points on a circle of radius r_sphere around (cx, cy, cz)
                    x_circle = cx + r_sphere * np.cos(theta)
                    y_circle = cy + r_sphere * np.sin(theta)
                    
                    # For each (x, y), solve for z on ellipsoid: z = ±c*sqrt(1 - x^2/a^2 - y^2/b^2)
                    z_circle = []
                    x_valid = []
                    y_valid = []
                    
                    for x, y in zip(x_circle, y_circle):
                        factor = 1.0 - (x/a_ellipsoid)**2 - (y/b_ellipsoid)**2
                        if factor > 0:
                            z_ellipsoid = c_ellipsoid * np.sqrt(factor)
                            # Choose sign of z to match sphere center
                            if cz > 0:
                                z_val = z_ellipsoid
                            else:
                                z_val = -z_ellipsoid
                            
                            # Verify this point is approximately on sphere boundary
                            dist_to_sphere_center = np.sqrt((x-cx)**2 + (y-cy)**2 + (z_val-cz)**2)
                            if abs(dist_to_sphere_center - r_sphere) < 0.2:  # tolerance
                                z_circle.append(z_val)
                                x_valid.append(x)
                                y_valid.append(y)
                    
                    if len(z_circle) > 0:
                        ax.plot(x_valid, y_valid, z_circle, color=obs_color, linewidth=3.5, alpha=0.8)
            
            # For cylinder constraint with rectangular box obstacle
            elif constraint_name.lower() == 'cylinder' and obstacle_box is not None:
                # Draw wireframe box obstacle
                x_half = obstacle_box['x_half']
                y_half = obstacle_box['y_half']
                z_half = obstacle_box['z_half']
                center = obstacle_box['center']
                
                # Define the 8 corners of the box
                corners = np.array([
                    [-x_half, -y_half, -z_half],
                    [x_half, -y_half, -z_half],
                    [x_half, y_half, -z_half],
                    [-x_half, y_half, -z_half],
                    [-x_half, -y_half, z_half],
                    [x_half, -y_half, z_half],
                    [x_half, y_half, z_half],
                    [-x_half, y_half, z_half]
                ]) + np.array(center)
                
                # Draw edges of the box
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
                ]
                
                for edge in edges:
                    pts = corners[list(edge)]
                    ax.plot3D(*pts.T, color=obs_color, linewidth=3.5, alpha=0.8)
            
            # For ellipsoid constraint with ellipsoid obstacle
            elif constraint_name.lower() == 'ellipsoid' and obstacle_semi_axes is not None:
                # Draw the ellipsoid obstacle as wireframe
                u = np.linspace(0, 2*np.pi, 30)
                v = np.linspace(0, np.pi, 20)
                a, b, c = obstacle_semi_axes[0], obstacle_semi_axes[1], obstacle_semi_axes[2]
                x_ell = a * np.outer(np.cos(u), np.sin(v))
                y_ell = b * np.outer(np.sin(u), np.sin(v))
                z_ell = c * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_wireframe(x_ell, y_ell, z_ell, color=obs_color, alpha=0.4, linewidth=2.0)

        # Optionally plot learned implicit surface if provided (QP mode)
        if learned_h_fn is not None and not combine_mpc_comparison and method == 'QP':
            verts_l, faces_l = compute_isosurface(learned_h_fn, bounds=bounds, res=res)
            if verts_l is not None and faces_l is not None:
                try:
                    ax.plot_trisurf(verts_l[:, 0], verts_l[:, 1], verts_l[:, 2], triangles=faces_l, alpha=0.35, color=learned_surface_color)
                except Exception:
                    pass

        linestyles = {'Imitation': '--', 'QP': '--', 'MPC': '-.', 'MPC (no obstacle)': '-', 'MPC (with obstacle)': '--', 'Learned': '--'}

        for idx in range(len(real_trajectories)):
            real_traj = np.asarray(real_trajectories[idx])
            color = colors[idx % len(colors)]

            # Plot real trajectory (solid) with modest transparency to reveal overlaps
            # Skip plotting real trajectory when comparing MPC with/without obstacle
            if not combine_mpc_comparison:
                # Use custom label if provided, otherwise use 'Real {idx+1}'
                traj_label = f'{real_label} {idx+1}' if len(real_trajectories) > 1 else real_label
                ax.plot(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], '-', color=color, linewidth=3.5, alpha=0.6, label=traj_label)
            
            # For combined MPC comparison, plot both versions
            if combine_mpc_comparison:
                # Plot MPC without obstacle
                mpc_no_obs_list = learned_trajectories.get('MPC (no obstacle)', [])
                if mpc_no_obs_list is not None and idx < len(mpc_no_obs_list):
                    traj_no_obs = np.asarray(mpc_no_obs_list[idx])
                    ax.plot(traj_no_obs[:, 0], traj_no_obs[:, 1], traj_no_obs[:, 2], '-', 
                           color=color, linewidth=3.5, alpha=0.6, label=f'MPC no-obs {idx+1}')
                
                # Plot MPC with obstacle
                mpc_with_obs_list = learned_trajectories.get('MPC (with obstacle)', [])
                if mpc_with_obs_list is not None and idx < len(mpc_with_obs_list):
                    traj_with_obs = np.asarray(mpc_with_obs_list[idx])
                    ax.plot(traj_with_obs[:, 0], traj_with_obs[:, 1], traj_with_obs[:, 2], '--', 
                           color=color, linewidth=3.5, alpha=0.7, label=f'MPC w/obs {idx+1}')
            else:
                # Normal case: single learned trajectory
                learned_traj = None
                if multi_methods:
                    learned_list = learned_trajectories.get(method, [])
                    if learned_list is not None and idx < len(learned_list):
                        learned_traj = np.asarray(learned_list[idx])
                else:
                    if learned_trajectories is not None and idx < len(learned_trajectories):
                        learned_traj = np.asarray(learned_trajectories[idx])

                if learned_traj is not None:
                    ls = linestyles.get(method, '--')
                    # Use custom learned label if provided, otherwise use display_name
                    if learned_label is not None:
                        label_text = f'{learned_label} {idx+1}' if len(real_trajectories) > 1 else learned_label
                    else:
                        label_text = f'{display_name} {idx+1}'
                    # Use more transparency for learned traces so overlaps with Real are visible
                    ax.plot(learned_traj[:, 0], learned_traj[:, 1], learned_traj[:, 2], ls, color=color, linewidth=3.5, alpha=0.45, label=label_text)

            # start/end markers - use MPC trajectory if in comparison mode, otherwise real trajectory
            if combine_mpc_comparison:
                # Use MPC with obstacle trajectory for markers
                mpc_with_obs_list = learned_trajectories.get('MPC (with obstacle)', [])
                if mpc_with_obs_list is not None and idx < len(mpc_with_obs_list):
                    ref_traj_for_markers = np.asarray(mpc_with_obs_list[idx])
                    ax.scatter(ref_traj_for_markers[0, 0], ref_traj_for_markers[0, 1], ref_traj_for_markers[0, 2], s=120, marker='o', color=color, alpha=0.8)
                    ax.scatter(ref_traj_for_markers[-1, 0], ref_traj_for_markers[-1, 1], ref_traj_for_markers[-1, 2], s=120, marker='s', color=color, alpha=0.8)
            else:
                ax.scatter(real_traj[0, 0], real_traj[0, 1], real_traj[0, 2], s=120, marker='o', color=color, alpha=0.8)
                ax.scatter(real_traj[-1, 0], real_traj[-1, 1], real_traj[-1, 2], s=120, marker='s', color=color, alpha=0.8)

        # Set axis limits based on constraint type and data
        if constraint_name.lower() == 'cylinder':
            # For cylinder: use fixed Z limits to show full height
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_zlim(-2.3, 2.3)
        else:
            # For other constraints: compute tight limits from trajectory data
            # Exception: for 'Learned' method (sinusoidal), use fixed bounds
            if method == 'Learned':
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
                ax.set_zlim(bounds[4], bounds[5])
            else:
                ax.set_xlim(bounds[0], bounds[1])
                ax.set_ylim(bounds[2], bounds[3])
                ax.set_zlim(bounds[4], bounds[5])
                try:
                    pts_list = []
                    for rt in real_trajectories:
                        rt = np.asarray(rt)
                        if rt.ndim == 2 and rt.size > 0:
                            pts_list.append(rt)
                    if multi_methods:
                        for lst in learned_trajectories.values():
                            if lst is not None:
                                for lt in lst:
                                    lt = np.asarray(lt)
                                    if lt.ndim == 2 and lt.size > 0:
                                        pts_list.append(lt)
                    else:
                        if learned_trajectories is not None:
                            for lt in learned_trajectories:
                                lt = np.asarray(lt)
                                if lt.ndim == 2 and lt.size > 0:
                                    pts_list.append(lt)
                    if len(pts_list) > 0:
                        all_pts = np.vstack(pts_list)
                        mins = np.min(all_pts, axis=0)
                        maxs = np.max(all_pts, axis=0)
                        ranges = maxs - mins
                        # add 8% margin or small absolute margin
                        margin = np.maximum(ranges * 0.08, 0.1)
                        ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
                        ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
                        ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
                except Exception:
                    pass
        
        # Set aspect ratio to show real proportions of the constraint surface
        try:
            # Get actual data ranges to compute proper aspect ratio
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            z_range = zlim[1] - zlim[0]
            
            # Set aspect ratio based on actual data ranges
            max_range = max(x_range, y_range, z_range)
            ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
        except Exception:
            # Fallback to equal aspect if set_box_aspect not available
            try:
                ax.set_aspect('equal')
            except Exception:
                pass
        
        # show grid for better readability
        ax.grid(True)
        ax.set_xlabel('X', fontsize=16, labelpad=10)
        ax.set_ylabel('Y', fontsize=16, labelpad=10)
        ax.set_zlabel('Z', fontsize=16, labelpad=10)
        ax.tick_params(labelsize=14)
        
        # Set tick spacing based on constraint type
        if constraint_name.lower() in ['sphere', 'ellipsoid']:
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.zaxis.set_major_locator(MultipleLocator(0.5))
        
        if combine_mpc_comparison:
            ax.set_title(f'Real vs MPC (with/without obstacle) — {constraint_name}', fontsize=18)
        else:
            ax.set_title(f'Real vs {display_name} — {constraint_name} (all trajectories)', fontsize=18)
        
        ax.legend(fontsize=14)
        ax.view_init(elev=35, azim=45)

        plt.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            if combine_mpc_comparison:
                out_path = f"{base}_MPC_comparison{ext}"
            else:
                out_path = f"{base}_{method}{ext}"
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f'Figure saved to: {out_path}')
        if show:
            plt.show()
        else:
            plt.close(fig)

        figs.append(fig)

    return figs


def plot_gt_surface_with_trajectories(h_fn, trajectories, save_path=None, show=True,
                                      bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48):
    """Plot GT isosurface (h_fn==0) and overlay training trajectories.

    Args:
        h_fn: callable that accepts (N,3) or single (3,) array and returns scalar(s)
        trajectories: list of trajectories. Each item can be:
            - an (N,3) array of points, or
            - a tuple/list (start_pt, goal_pt, trajectory_array)
        save_path: optional path to save the figure
        show: whether to display the figure
        bounds: plotting bounds [x_min,x_max,y_min,y_max,z_min,z_max]
        res: grid resolution per axis for marching cubes
    """
    import matplotlib.pyplot as plt
    try:
        from skimage.measure import marching_cubes
    except Exception:
        marching_cubes = None

    # Prepare trajectories as arrays
    traj_arrays = []
    for item in trajectories:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                traj = np.asarray(item[2])
            else:
                traj = np.asarray(item)
            if traj.ndim == 1 and traj.size % 3 == 0:
                traj = traj.reshape(-1, 3)
            if traj.ndim == 2 and traj.shape[1] == 3:
                traj_arrays.append(traj.copy())
        except Exception:
            continue

    # Build grid
    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[2], bounds[3], res)
    z = np.linspace(bounds[4], bounds[5], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    # Evaluate GT function
    try:
        vals = h_fn(pts)
    except Exception:
        vals = np.array([h_fn(p) for p in pts])
    GT = vals.reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if marching_cubes is not None and np.any(GT <= 0.0) and np.any(GT >= 0.0):
        dx = (bounds[1] - bounds[0]) / (res - 1)
        dy = (bounds[3] - bounds[2]) / (res - 1)
        dz = (bounds[5] - bounds[4]) / (res - 1)
        try:
            verts, faces, _, _ = marching_cubes(GT, level=0.0, spacing=(dx, dy, dz))
            verts[:, 0] += bounds[0]
            verts[:, 1] += bounds[2]
            verts[:, 2] += bounds[4]
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, alpha=0.4, color='red')
        except Exception:
            # fallback: continue without surface
            pass
    else:
        # If marching_cubes unavailable or no zero-crossing, skip surface plotting
        pass

    # Overlay trajectories
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, traj in enumerate(traj_arrays):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color=colors[i % len(colors)], linewidth=1.5, alpha=0.9)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors[i % len(colors)], s=30)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors[i % len(colors)], s=30, marker='s')

    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_zlim(bounds[4], bounds[5])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('GT Surface + Training Trajectories')
    # tighten limits around trajectories and show grid
    try:
        if len(traj_arrays) > 0:
            all_pts = np.vstack(traj_arrays)
            mins = np.min(all_pts, axis=0)
            maxs = np.max(all_pts, axis=0)
            ranges = maxs - mins
            margin = np.maximum(ranges * 0.08, 0.1)
            ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
            ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
            ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
    except Exception:
        pass
    ax.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_projection_comparison(real_trajectories, nn_trajectories, constraint_name, qp_trajectories=None, mpc_trajectories=None,
                             mpc_trajectories_no_obs=None, save_path=None, show=True, h_fn=None,
                             bounds=[-2.5, 2.5, -2.5, 2.5], res=80):
    """
    Plot trajectories in a FIXED top-down XY view, regardless of trajectory tilt.
    Useful for consistent visual comparison across constraints.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Number of trajectories to display
    # If we have MPC with/without obstacle, use those to determine count
    if mpc_trajectories is not None and mpc_trajectories_no_obs is not None:
        n = min(3, len(mpc_trajectories))
    else:
        n = min(3, len(real_trajectories)) if len(real_trajectories) > 0 else 0
    
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]

    x_min, x_max, y_min, y_max = bounds

    for i in range(n):
        ax = axes[i]

        # For MPC obstacle comparison mode, skip real trajectories
        show_real = len(real_trajectories) > 0 and i < len(real_trajectories)
        real_traj = np.asarray(real_trajectories[i]) if show_real else None
        
        def _valid_traj(obj):
            if obj is None:
                return None
            try:
                arr = np.asarray(obj)
            except Exception:
                return None
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 2:
                return arr
            return None

        nn_traj = _valid_traj(nn_trajectories[i]) if (nn_trajectories is not None and i < len(nn_trajectories)) else None
        qp_traj = _valid_traj(qp_trajectories[i]) if (qp_trajectories is not None and i < len(qp_trajectories)) else None
        mpc_traj = _valid_traj(mpc_trajectories[i]) if (mpc_trajectories is not None and i < len(mpc_trajectories)) else None
        mpc_traj_no_obs = _valid_traj(mpc_trajectories_no_obs[i]) if (mpc_trajectories_no_obs is not None and i < len(mpc_trajectories_no_obs)) else None

        # ----------------------------------------
        # Plot projected trajectories (just XY)
        colors = {'Real': 'k', 'Imitation': 'C0', 'QP': 'C1', 'MPC': 'C2', 'MPC_no_obs': 'C3'}

        # Skip plotting real trajectory when comparing MPC with/without obstacle
        if real_traj is not None and mpc_traj_no_obs is None:
            ax.plot(real_traj[:, 0], real_traj[:, 1], '-k', linewidth=3.5, label="Real")

        if nn_traj is not None:
            ax.plot(nn_traj[:, 0], nn_traj[:, 1], '--', color=colors['Imitation'], linewidth=3.0, label="Imitation")

        if qp_traj is not None:
            ax.plot(qp_traj[:, 0], qp_traj[:, 1], '--', color=colors['QP'], linewidth=3.0, label="projection")

        if mpc_traj_no_obs is not None:
            ax.plot(mpc_traj_no_obs[:, 0], mpc_traj_no_obs[:, 1], '-k', color=colors['MPC_no_obs'], linewidth=3.5, label="MPC (no obstacle)")

        if mpc_traj is not None:
            ax.plot(mpc_traj[:, 0], mpc_traj[:, 1], '--', color=colors['MPC'], linewidth=3.5, label="MPC (with obstacle)")

        # Start / end markers - use MPC trajectory if no real trajectory
        ref_traj = real_traj if real_traj is not None else (mpc_traj if mpc_traj is not None else None)
        if ref_traj is not None:
            ax.scatter(ref_traj[0, 0], ref_traj[0, 1], s=120, color='green')
            ax.scatter(ref_traj[-1, 0], ref_traj[-1, 1], s=120, color='red')

        # Compute tight XY limits from available trajectories
        try:
            pts2 = []
            if real_traj is not None:
                pts2.append(real_traj[:, :2])
            if nn_traj is not None:
                pts2.append(nn_traj[:, :2])
            if qp_traj is not None:
                pts2.append(qp_traj[:, :2])
            if mpc_traj is not None:
                pts2.append(mpc_traj[:, :2])
            all2 = np.vstack([p for p in pts2 if p is not None and p.size > 0])
            mins = np.min(all2, axis=0)
            maxs = np.max(all2, axis=0)
            ranges = maxs - mins
            margin = np.maximum(ranges * 0.08, 0.05)
            ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
            ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
        except Exception:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{constraint_name} – Trajectory {i+1}", fontsize=16)
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print("Saved to:", save_path)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_trajectory_comparison_MPC_animated(real_trajectories, learned_trajectories, constraint_name, save_path=None,
                               h_fn=None, bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], res=48,
                               surface_alpha=0.4, surface_color='red', learned_h_fn=None, learned_surface_color='cyan',
                               h_obs_fn=None, obs_color='orange', obs_alpha=0.3, obstacle_radius=0.5, obstacle_semi_axes=None,
                               obstacle_spheres=None, obstacle_plane=None, obstacle_box=None, interval=50, 
                               reference_label='Reference trajectory', animate_reference=True):
    """Animated version of trajectory comparison showing MPC trajectories being drawn progressively.
    
    Args:
        real_trajectories: List of real trajectory arrays (max 3)
        learned_trajectories: Dict mapping method name -> list of arrays (must contain 'MPC (with obstacle)')
        constraint_name: Name of constraint
        save_path: Optional path to save animation (as .gif or .mp4)
        interval: Animation frame interval in milliseconds (default: 50)
        reference_label: Label for reference trajectories (default: 'Reference trajectory')
        animate_reference: If True, animate reference trajectories; if False, show them static (default: True)
        Other args: Same as plot_trajectory_comparison
    """
    from matplotlib.animation import FuncAnimation
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Extract MPC trajectories - try all possible MPC keys
    mpc_trajectories = None
    mpc_method_name = None
    
    for key in learned_trajectories.keys():
        if 'MPC' in key and learned_trajectories[key]:
            mpc_trajectories = learned_trajectories[key]
            mpc_method_name = key
            break
    
    if not mpc_trajectories:
        print("No MPC trajectories found for animation")
        return None
    
    print(f"Animating {len(mpc_trajectories)} trajectories for method: {mpc_method_name}")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot GT isosurface if h_fn is provided
    if h_fn is not None:
        verts, faces = compute_isosurface(h_fn, bounds=bounds, res=res)
        if verts is not None and faces is not None:
            try:
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, alpha=surface_alpha, color='lightcoral')
            except Exception:
                pass
    
    # Plot obstacle visualization (same as in plot_trajectory_comparison)
    if h_obs_fn is not None and h_fn is not None:
        theta = np.linspace(0, 2*np.pi, 100)
        
        if constraint_name.lower() == 'sphere' and obstacle_semi_axes is None and obstacle_spheres is None:
            sphere_radius = 1.0
            num_lines = 16
            theta_lines = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
            z_range = np.linspace(-sphere_radius, sphere_radius, 50)
            
            for angle in theta_lines:
                x_line = obstacle_radius * np.cos(angle)
                y_line = obstacle_radius * np.sin(angle)
                x_arr = np.full_like(z_range, x_line)
                y_arr = np.full_like(z_range, y_line)
                ax.plot(x_arr, y_arr, z_range, color=obs_color, linewidth=1.5, alpha=0.6)
            
            theta_circle = np.linspace(0, 2*np.pi, 100)
            x_circle = obstacle_radius * np.cos(theta_circle)
            y_circle = obstacle_radius * np.sin(theta_circle)
            
            z_top = np.full_like(x_circle, sphere_radius)
            ax.plot(x_circle, y_circle, z_top, color=obs_color, linewidth=3.5, alpha=0.8)
            
            z_bottom = np.full_like(x_circle, -sphere_radius)
            ax.plot(x_circle, y_circle, z_bottom, color=obs_color, linewidth=3.5, alpha=0.8)
        
        elif constraint_name.lower() == 'ellipsoid' and obstacle_semi_axes is None and obstacle_spheres is None:
            a_ellipsoid, b_ellipsoid, c_ellipsoid = 0.8, 1.2, 1.5
            num_lines = 16
            theta_lines = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
            z_range = np.linspace(-c_ellipsoid, c_ellipsoid, 50)
            
            for angle in theta_lines:
                x_line = obstacle_radius * np.cos(angle)
                y_line = obstacle_radius * np.sin(angle)
                x_arr = np.full_like(z_range, x_line)
                y_arr = np.full_like(z_range, y_line)
                ax.plot(x_arr, y_arr, z_range, color=obs_color, linewidth=2.5, alpha=0.6)
            
            theta_circle = np.linspace(0, 2*np.pi, 100)
            x_circle = obstacle_radius * np.cos(theta_circle)
            y_circle = obstacle_radius * np.sin(theta_circle)
            
            z_top = np.full_like(x_circle, c_ellipsoid)
            ax.plot(x_circle, y_circle, z_top, color=obs_color, linewidth=3.5, alpha=0.8)
            
            z_bottom = np.full_like(x_circle, -c_ellipsoid)
            ax.plot(x_circle, y_circle, z_bottom, color=obs_color, linewidth=3.5, alpha=0.8)
    
    # Initialize line objects for each trajectory
    n_trajs = len(mpc_trajectories)
    line_objects = []
    marker_objects = []
    real_line_objects = []
    real_moving_points = []  # Moving points for real trajectories
    mpc_moving_points = []    # Moving points for MPC trajectories
    
    # First, plot real trajectories - animated or static based on parameter
    for idx in range(min(n_trajs, len(real_trajectories))):
        color = colors[idx % len(colors)]
        real_traj = np.asarray(real_trajectories[idx])
        
        if animate_reference:
            # Create empty line for real trajectory (will be animated)
            real_line, = ax.plot([], [], [], '-', color=color, linewidth=3.5, alpha=0.4, 
                                label=reference_label)
            real_line_objects.append(real_line)
            
            # Create moving point for real trajectory
            real_point = ax.scatter([], [], [], s=150, marker='o', color=color, alpha=0.9, edgecolors='black', linewidths=2)
            real_moving_points.append(real_point)
        else:
            # Plot complete static reference trajectory
            real_line, = ax.plot(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], 
                                '-', color=color, linewidth=3.5, alpha=0.4, 
                                label=reference_label)
            real_line_objects.append(real_line)
            # No moving point for static trajectories
    
    # Then create animated MPC trajectories - labeled by the method name key
    for idx in range(n_trajs):
        color = colors[idx % len(colors)]
        traj = np.asarray(mpc_trajectories[idx])
        
        # Create empty line for MPC with higher opacity - use the method name from the dict key
        line, = ax.plot([], [], [], '--', color=color, linewidth=3.5, alpha=0.95, label=f'{mpc_method_name}')
        line_objects.append(line)
        
        # Start and end markers
        start_marker = ax.scatter([], [], [], s=120, marker='o', color=color, alpha=0.8)
        end_marker = ax.scatter([], [], [], s=120, marker='s', color=color, alpha=0.8)
        marker_objects.append((start_marker, end_marker))
        
        # Create moving point for MPC trajectory
        mpc_point = ax.scatter([], [], [], s=150, marker='o', color=color, alpha=0.9, edgecolors='black', linewidths=2)
        mpc_moving_points.append(mpc_point)
    
    # Find max trajectory length for animation (consider both real and MPC trajectories)
    max_len = max(len(np.asarray(t)) for t in mpc_trajectories)
    if real_trajectories:
        max_real_len = max(len(np.asarray(t)) for t in real_trajectories if len(np.asarray(t)) > 0)
        max_len = max(max_len, max_real_len)
    
    def init():
        # Initialize real trajectory lines and points (only if animating)
        if animate_reference:
            for line in real_line_objects:
                line.set_data([], [])
                line.set_3d_properties([])
            for point in real_moving_points:
                point._offsets3d = ([], [], [])
        
        # Initialize MPC trajectory lines and points
        for line in line_objects:
            line.set_data([], [])
            line.set_3d_properties([])
        for start_m, end_m in marker_objects:
            start_m._offsets3d = ([], [], [])
            end_m._offsets3d = ([], [], [])
        for point in mpc_moving_points:
            point._offsets3d = ([], [], [])
        
        return (real_line_objects + real_moving_points + line_objects + 
                [m for pair in marker_objects for m in pair] + mpc_moving_points)
    
    def animate(frame):
        # Animate real trajectories (only if enabled)
        if animate_reference:
            for idx, line in enumerate(real_line_objects):
                if idx < len(real_trajectories):
                    real_traj = np.asarray(real_trajectories[idx])
                    end_idx = min(frame + 1, len(real_traj))
                    
                    if end_idx > 0:
                        line.set_data(real_traj[:end_idx, 0], real_traj[:end_idx, 1])
                        line.set_3d_properties(real_traj[:end_idx, 2])
                        
                        # Update moving point for real trajectory
                        point = real_moving_points[idx]
                        point._offsets3d = ([real_traj[end_idx-1, 0]], 
                                           [real_traj[end_idx-1, 1]], 
                                           [real_traj[end_idx-1, 2]])
        
        # Animate MPC trajectories
        for idx, line in enumerate(line_objects):
            traj = np.asarray(mpc_trajectories[idx])
            # Show trajectory up to current frame
            end_idx = min(frame + 1, len(traj))
            
            if end_idx > 0:
                line.set_data(traj[:end_idx, 0], traj[:end_idx, 1])
                line.set_3d_properties(traj[:end_idx, 2])
                
                # Update moving point for MPC trajectory
                point = mpc_moving_points[idx]
                point._offsets3d = ([traj[end_idx-1, 0]], 
                                   [traj[end_idx-1, 1]], 
                                   [traj[end_idx-1, 2]])
                
                # Update markers
                start_m, end_m = marker_objects[idx]
                start_m._offsets3d = ([traj[0, 0]], [traj[0, 1]], [traj[0, 2]])
                
                # Only show end marker when trajectory is complete
                if end_idx == len(traj):
                    end_m._offsets3d = ([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]])
                else:
                    end_m._offsets3d = ([], [], [])
        
        return (real_line_objects + real_moving_points + line_objects + 
                [m for pair in marker_objects for m in pair] + mpc_moving_points)
    
    # Calculate tight bounds from trajectory data (like in plot_trajectory_comparison)
    try:
        pts_list = []
        for rt in real_trajectories:
            rt = np.asarray(rt)
            if rt.ndim == 2 and rt.size > 0:
                pts_list.append(rt)
        for traj in mpc_trajectories:
            traj = np.asarray(traj)
            if traj.ndim == 2 and traj.size > 0:
                pts_list.append(traj)
        
        if len(pts_list) > 0:
            all_pts = np.vstack(pts_list)
            mins = np.min(all_pts, axis=0)
            maxs = np.max(all_pts, axis=0)
            ranges = maxs - mins
            # add 8% margin or small absolute margin
            margin = np.maximum(ranges * 0.08, 0.1)
            ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
            ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
            ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
        else:
            # Fallback to provided bounds
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_zlim(bounds[4], bounds[5])
    except Exception:
        # Fallback to provided bounds
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_zlim(bounds[4], bounds[5])
    
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
        try:
            ax.set_aspect('equal')
        except Exception:
            pass
    
    ax.grid(False)
    ax.set_axis_off()  # Eliminar eixos
    
    # Fons blanc
    fig.patch.set_facecolor('white')
    ax.patch.set_facecolor('white')
    
    ax.set_title(f'MPC Trajectories — {constraint_name} (animated)', fontsize=18)
    ax.legend(fontsize=14)
    ax.view_init(elev=75, azim=45)
    
    plt.tight_layout()
    
    # Create animation (use fewer frames for faster generation)
    num_frames = max(max_len // 2, 30)  # Show every 2nd point, minimum 30 frames
    frame_step = max(max_len // num_frames, 1)
    frames = range(0, max_len, frame_step)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, 
                        interval=interval, blit=True, repeat=True)
    
    # Save animation if path provided (MUST save before showing!)
    if save_path:
        print(f'Saving animation to: {save_path}')
        print('This may take a few moments...')
        try:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval, dpi=100)
                print(f'✓ Animation (GIF) saved successfully to: {save_path}')
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=100)
                print(f'✓ Animation (MP4) saved successfully to: {save_path}')
            else:
                print('Warning: Save path must end with .gif or .mp4')
        except Exception as e:
            print(f'Error saving animation: {e}')
    
    # Close the figure instead of showing it (animation is already saved)
    plt.close(fig)
    
    return fig, anim

