"""
3D-specific visualization utilities for implicit constraint learning models.
Contains only essential functions: isosurface_with_samples_grid and projection_visualization_3d
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes

def plot_GT_with_samples_grid(
    gt_samples, pos_np, neg_np,
    bounds=[-2, 2, -2, 2, 0, 4], level=0.0, 
    constraint_name=None, save_path=None, gt_function=None, show_full_surface=False
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Will set aspect ratio after plotting data
    n_neg_show = len(neg_np)
    n_pos_show = len(pos_np)
    n_gt_show  = len(gt_samples)

    rng = np.random.default_rng(42)
    neg_idx = rng.choice(len(neg_np), n_neg_show, replace=False) if len(neg_np) else []
    pos_idx = rng.choice(len(pos_np), n_pos_show, replace=False) if len(pos_np) else []
    gt_idx  = rng.choice(len(gt_samples), n_gt_show, replace=False) if len(gt_samples) else []

    neg_show = neg_np[neg_idx] if len(neg_np) else np.empty((0,3))
    pos_show = pos_np[pos_idx] if len(pos_np) else np.empty((0,3))
    gt_show  = gt_samples[gt_idx] if len(gt_samples) else np.empty((0,3))

    # Apply hemisphere filter for sphere and ellipsoid constraints (unless full surface requested)
    if constraint_name in ["sphere", "ellipsoid"] and not show_full_surface:
        if len(neg_show):
            neg_show = neg_show[neg_show[:, 1] >= 0]
        if len(pos_show):
            pos_show = pos_show[pos_show[:, 1] >= 0]
        if len(gt_show):
            gt_show = gt_show[gt_show[:, 1] >= 0]

    # For sphere and ellipsoid constraints we want to show negatives as "inside" vs "outside"
    # based on the GT function. For other constraints (hyperplane, spiral, ...)
    # there is no semantic inside/outside, so plot negatives with a single color.
    if len(neg_show):
        if constraint_name in ["sphere", "ellipsoid"] and gt_function is not None:
            neg_f_vals = np.array([gt_function(p) for p in neg_show]).flatten()
            neg_inside = neg_show[neg_f_vals < 0]
            neg_outside = neg_show[neg_f_vals >= 0]

            if len(neg_inside):
                ax.scatter(neg_inside[:,0], neg_inside[:,1], neg_inside[:,2],
                           s=60, alpha=0.8, c='lightgreen', label="Neg Inside")
            if len(neg_outside):
                ax.scatter(neg_outside[:,0], neg_outside[:,1], neg_outside[:,2],
                           s=60, alpha=0.8, c='orange', label="Neg Outside")
        else:
            ax.scatter(neg_show[:,0], neg_show[:,1], neg_show[:,2],
                       s=60, alpha=0.8, c='orange', label="Negative")
    
    if len(pos_show):
        ax.scatter(pos_show[:,0], pos_show[:,1], pos_show[:,2],
                   s=100, alpha=0.8, c='blue', label="Positive")

    # Special handling for spiral (1D curve) vs surfaces (sphere, hyperplane, spiral_tube, etc.)
    # Note: spiral_tube is a surface, not a curve
    if constraint_name == "spiral":
        # For spiral: draw GT as continuous line, not marching cubes
        if len(gt_samples) > 0:
            ax.plot(gt_samples[:,0], gt_samples[:,1], gt_samples[:,2],
                   'r-', linewidth=5.0, alpha=0.9, label="GT Spiral", zorder=10)
    elif gt_function is not None:
        try:
            from skimage.measure import marching_cubes

            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            gt_resolution = 30
            x = np.linspace(x_min, x_max, gt_resolution)
            y = np.linspace(y_min, y_max, gt_resolution)
            z = np.linspace(z_min, z_max, gt_resolution)

            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

            gt_values = np.asarray([gt_function(p) for p in pts], dtype=float).reshape(X.shape)

            if np.any(gt_values <= level) and np.any(gt_values >= level):
                dx = (x_max - x_min) / (gt_resolution - 1)
                dy = (y_max - y_min) / (gt_resolution - 1)
                dz = (z_max - z_min) / (gt_resolution - 1)

                verts, faces, _, _ = marching_cubes(
                    gt_values, level=level, spacing=(dx, dy, dz)
                )
                verts[:, 0] += x_min
                verts[:, 1] += y_min
                verts[:, 2] += z_min
                
                # Apply hemisphere filter for surface vertices for sphere and ellipsoid (unless full surface requested)
                if constraint_name in ["sphere", "ellipsoid"] and not show_full_surface:
                    mask = verts[:, 1] >= 0
                    vert_map = np.full(len(verts), -1, dtype=int)
                    vert_map[mask] = np.arange(np.sum(mask))
                    
                    verts_filtered = verts[mask]
                    
                    faces_valid = np.all(mask[faces], axis=1)
                    faces_filtered = faces[faces_valid]
                    faces_filtered = vert_map[faces_filtered]
                else:
                    verts_filtered = verts
                    faces_filtered = faces
                
                if len(verts_filtered) > 0 and len(faces_filtered) > 0:
                    ax.plot_trisurf(
                        verts_filtered[:, 0], verts_filtered[:, 1], verts_filtered[:, 2],
                        triangles=faces_filtered, alpha=0.4, color='lightcoral', label="GT Surface"
                    )
                # expose verts for adaptive bounds computation
                verts_for_bounds = verts_filtered if 'verts_filtered' in locals() else None
            else:
                if len(gt_show):
                    ax.scatter(gt_show[:,0], gt_show[:,1], gt_show[:,2],
                               s=10, alpha=0.5, c='red', label="GT")
        except Exception as e:
            if len(gt_show):
                ax.scatter(gt_show[:,0], gt_show[:,1], gt_show[:,2],
                           s=10, alpha=0.5, c='red', label="GT")
    else:
        if len(gt_show):
            ax.scatter(gt_show[:,0], gt_show[:,1], gt_show[:,2],
                       s=10, alpha=0.5, c='red', label="GT")

    # Compute adaptive bounds from available plotted data (pos, neg, gt samples, surface verts)
    try:
        pts_collection = []
        if len(pos_show):
            pts_collection.append(pos_show)
        if len(neg_show):
            pts_collection.append(neg_show)
        if len(gt_show):
            pts_collection.append(gt_show)
        # include marching-cubes vertices if available
        if 'verts_for_bounds' in locals() and verts_for_bounds is not None and getattr(verts_for_bounds, 'size', 0) > 0:
            pts_collection.append(np.asarray(verts_for_bounds))

        if len(pts_collection) > 0:
            all_pts = np.vstack(pts_collection)
            mins = np.min(all_pts, axis=0)
            maxs = np.max(all_pts, axis=0)
            ranges = maxs - mins
            margin = np.maximum(ranges * 0.08, 0.1)
            ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
            ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
            ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])
        else:
            ax.set_xlim([bounds[0], bounds[1]])
            ax.set_ylim([bounds[2], bounds[3]])
            ax.set_zlim([bounds[4], bounds[5]])
    except Exception:
        ax.set_xlim([bounds[0], bounds[1]])
        ax.set_ylim([bounds[2], bounds[3]])
        ax.set_zlim([bounds[4], bounds[5]])
    
    # Set aspect ratio based on actual axis ranges
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
    

    # Set axis ticks every 0.5 units
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=18, labelpad=10)
    ax.set_ylabel('Y', fontsize=18, labelpad=10)
    ax.set_zlabel('Z', fontsize=18, labelpad=10)
    ax.tick_params(labelsize=16)
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left', fontsize=16)

    ax.set_title(f'3D GT + Samples - {constraint_name} - n_samples: {len(pos_np)}', fontsize=20, fontweight='bold')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()

def plot_GT_LS_Countours(
    results,
    bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    constraint_name=None,
    save_path=None,
    gt_function=None,
    level=0.0,
    resolution=30,
):
    def filter_mesh_halfspace(verts, faces, axis_idx=1, threshold=0.0, op=">="):
        if op == ">=":
            mask = verts[:, axis_idx] >= threshold
        else:
            mask = verts[:, axis_idx] <= threshold
        faces_kept_mask = np.all(mask[faces], axis=1)
        faces_kept = faces[faces_kept_mask]
        if faces_kept.size == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)
        old_to_new = -np.ones(len(verts), dtype=int)
        old_to_new[np.where(mask)[0]] = np.arange(mask.sum())
        faces_remap = old_to_new[faces_kept]
        return verts[mask], faces_remap
    
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Special handling for spiral constraint (1D curve)
    # Note: spiral_tube is a SURFACE (2D), not a curve (1D) - treat it as a surface!
    if constraint_name == "spiral":  # Only 1D spiral, NOT spiral_tube
        # For spiral: use gt_curve to get ordered points and draw as line
        from src.data.datasets3D import make_constraint_3d
        try:
            _, _, _, gt_curve = make_constraint_3d("spiral")
            spiral_pts = gt_curve(M=500)  # Dense sampling
            ax.plot(spiral_pts[:, 0], spiral_pts[:, 1], spiral_pts[:, 2],
                   'r-', linewidth=5.0, alpha=0.5, label='GT Spiral', zorder=10)
        except Exception as e:
            print(f"Warning: Could not create GT spiral line: {e}")
    elif gt_function is not None:
        try:
            gt_values = np.asarray([gt_function(p) for p in pts], dtype=float).reshape(X.shape)
            if np.any(gt_values <= level) and np.any(gt_values >= level):
                dx = (x_max - x_min) / (resolution - 1)
                dy = (y_max - y_min) / (resolution - 1)
                dz = (z_max - z_min) / (resolution - 1)

                verts, faces, _, _ = marching_cubes(gt_values, level=level, spacing=(dx, dy, dz))
                verts[:, 0] += x_min; verts[:, 1] += y_min; verts[:, 2] += z_min

              
                verts_f, faces_f = verts, faces

                if len(verts_f) and len(faces_f):
                    ax.plot_trisurf(verts_f[:, 0], verts_f[:, 1], verts_f[:, 2],
                                    triangles=faces_f, color='lightcoral', alpha=0.8, linewidth=0.4,
                                    label='GT Surface')
        except Exception as e:
            print(f"Warning: Could not create GT surface: {e}")

    colors = ['green', 'purple', 'cyan', 'magenta', 'yellow', 'pink']
    color_idx = 0

    for (arch_name, loss_name), result in results.items():
        model = result[0] if isinstance(result, (list, tuple)) else result
        if model is None:
            print(f"Warning: No model found for {arch_name} - {loss_name}")
            continue
        try:
            # ALL models use marching cubes for visualization consistency
            # (even spiral - we sample with sample_curve if available, but visualize with mesh)
            device = next(model.parameters()).device
            pts_torch = torch.tensor(pts, dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                pred_values = model(pts_torch).detach().cpu().numpy().reshape(X.shape)

            if np.any(pred_values <= level) and np.any(pred_values >= level):
                dx = (x_max - x_min) / (resolution - 1)
                dy = (y_max - y_min) / (resolution - 1)
                dz = (z_max - z_min) / (resolution - 1)

                verts, faces, _, _ = marching_cubes(pred_values, level=level, spacing=(dx, dy, dz))
                verts[:, 0] += x_min; verts[:, 1] += y_min; verts[:, 2] += z_min

             
                verts_f, faces_f = verts, faces

                if len(verts_f) and len(faces_f):
                    color = colors[color_idx % len(colors)]
                    ax.plot_trisurf(verts_f[:, 0], verts_f[:, 1], verts_f[:, 2],
                                    triangles=faces_f, edgecolor=color, alpha=0.35, linewidth=0.4,
                                    label=f'{arch_name} - {loss_name}')
                    color_idx += 1
            else:
                print(f"Warning: No valid isosurface found for {arch_name} - {loss_name}")
        except Exception as e:
            print(f"Warning: Could not create surface for {arch_name} - {loss_name}: {e}")
        
    ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max]); ax.set_zlim([z_min, z_max])
    ax.set_xlabel('X', fontsize=18, labelpad=10); ax.set_ylabel('Y', fontsize=18, labelpad=10); ax.set_zlabel('Z', fontsize=18, labelpad=10)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set aspect ratio based on actual data ranges
    try:
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
    except Exception:
        pass
    
    ax.view_init(elev=22, azim=45)
    ax.legend(loc='upper left', fontsize=14)
    
    # Clean up title: extract first loss_name from results for display
    # Remove redundant regularizer info from title
    display_loss_name = loss_name.replace(' (no reg)', '')
    ax.set_title(f'3D Surfaces Comparison - {constraint_name} - {display_loss_name}', fontsize=18, fontweight='bold')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

    return fig

def plot_projection_visualization_3d(projection_info, gt_samples, bounds=[-2, 2, -2, 2, 0, 4], 
                                   save_path=None, n_show=50, gt_function=None, constraint_name=None):
    """
    Visualize 3D projection pairs (x, x') using pre-computed projection information.
    Shows hemisphere (y >= 0) only for sphere constraint.
    """
    # Extract data from projection_info
    original_np = projection_info['original_points'].numpy()
    projected_np = projection_info['projected_points'].numpy()
    f_vals_np = projection_info['original_f_vals'].numpy()
    proj_f_vals_np = projection_info['projected_f_vals'].numpy()
    n_pos = projection_info['n_pos']
    n_neg = projection_info['n_neg']
    
    # Limit number of points to show
    n_pos_show = min(n_show, n_pos)
    n_neg_show = min(n_show, n_neg)
    
    # Split back into pos and neg
    pos_orig = original_np[:n_pos_show]
    pos_proj = projected_np[:n_pos_show]
    neg_orig = original_np[n_pos:n_pos+n_neg_show] if n_neg > 0 else np.empty((0, 3))
    neg_proj = projected_np[n_pos:n_pos+n_neg_show] if n_neg > 0 else np.empty((0, 3))
    
    # Apply hemisphere filter only for sphere constraint
    if constraint_name == "sphere":
        # HEMISPHERE FILTER: Keep only points with y >= 0 (both original and projected)
        if len(pos_orig) > 0:
            pos_mask = (pos_orig[:, 1] >= 0) & (pos_proj[:, 1] >= 0)
            pos_orig = pos_orig[pos_mask]
            pos_proj = pos_proj[pos_mask]
            n_pos_show = len(pos_orig)
        
        if len(neg_orig) > 0:
            neg_mask = (neg_orig[:, 1] >= 0) & (neg_proj[:, 1] >= 0)
            neg_orig = neg_orig[neg_mask]
            neg_proj = neg_proj[neg_mask]
            n_neg_show = len(neg_orig)
        
        # Filter gt_samples as well
        if len(gt_samples) > 0:
            gt_samples_filtered = gt_samples[gt_samples[:, 1] >= 0]
        else:
            gt_samples_filtered = gt_samples
    else:
        gt_samples_filtered = gt_samples
    
    fig = plt.figure(figsize=(15, 6))
    
    # Left plot: 3D Original vs Projected points
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot positive points and projections FIRST (so GT is drawn on top)
    if len(pos_orig) > 0:
        ax1.scatter(pos_orig[:, 0], pos_orig[:, 1], pos_orig[:, 2], 
                   c='blue', s=60, alpha=0.5, label='Pos Original', zorder=1)
        ax1.scatter(pos_proj[:, 0], pos_proj[:, 1], pos_proj[:, 2], 
                   c='lightblue', s=40, alpha=0.6, label='Pos Projected', zorder=1)
    
    # Plot negative points and projections  
    if len(neg_orig) > 0:
        ax1.scatter(neg_orig[:, 0], neg_orig[:, 1], neg_orig[:, 2], 
                   c='orange', s=60, alpha=0.5, label='Neg Original', zorder=1)
        ax1.scatter(neg_proj[:, 0], neg_proj[:, 1], neg_proj[:, 2], 
                   c='yellow', s=40, alpha=0.6, label='Neg Projected', zorder=1)
    
    # Draw projection lines for positive points
    n_lines_pos = min(30, n_pos_show)
    for i in range(n_lines_pos):
        if i < len(pos_orig):
            ax1.plot([pos_orig[i, 0], pos_proj[i, 0]], 
                    [pos_orig[i, 1], pos_proj[i, 1]], 
                    [pos_orig[i, 2], pos_proj[i, 2]], 
                    'b-', alpha=0.3, linewidth=2.0, zorder=1)
    
    # Draw projection lines for negative points
    n_lines_neg = min(30, n_neg_show)
    for i in range(n_lines_neg):
        if i < len(neg_orig):
            ax1.plot([neg_orig[i, 0], neg_proj[i, 0]], 
                    [neg_orig[i, 1], neg_proj[i, 1]], 
                    [neg_orig[i, 2], neg_proj[i, 2]], 
                    color='darkorange', alpha=0.3, linewidth=2.0, zorder=1)
    
    # NOW plot GT on top with high zorder
    # Special handling for spiral constraint (1D curve)
    # Note: spiral_tube is a surface, not a curve
    if constraint_name == "spiral":
        # For spiral: draw GT as THICK continuous line ON TOP
        if len(gt_samples) > 0:
            ax1.plot(gt_samples[:, 0], gt_samples[:, 1], gt_samples[:, 2],
                    'r-', linewidth=5.0, alpha=1.0, label='GT Spiral', zorder=100)
    elif gt_function is not None:
        # Plot ground truth surface (continuous) if function is provided
        try:
            from skimage import measure
            
            # Create a 3D grid for isosurface
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            resolution = 20
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            z = np.linspace(z_min, z_max, resolution)
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
            
            # Evaluate GT function
            gt_values = np.array([gt_function(point) for point in points])
            gt_values = gt_values.reshape(X.shape)
            
            # Extract isosurface using marching cubes
            if np.any(gt_values <= 0.0) and np.any(gt_values >= 0.0):
                verts, faces, _, _ = measure.marching_cubes(gt_values, level=0.0, spacing=(
                    (x_max-x_min)/(resolution-1), 
                    (y_max-y_min)/(resolution-1), 
                    (z_max-z_min)/(resolution-1)
                ))
                # Adjust vertices to match coordinate system
                verts[:, 0] += x_min
                verts[:, 1] += y_min  
                verts[:, 2] += z_min
                
                # Apply hemisphere filter for sphere and ellipsoid constraints
                if constraint_name in ["sphere", "ellipsoid"]:
                    mask = verts[:, 1] >= 0
                    vert_map = np.full(len(verts), -1, dtype=int)
                    vert_map[mask] = np.arange(np.sum(mask))
                    
                    verts_filtered = verts[mask]
                    
                    # Keep only faces where all vertices have y >= 0
                    faces_valid = np.all(mask[faces], axis=1)
                    faces_filtered = faces[faces_valid]
                    faces_filtered = vert_map[faces_filtered]
                else:
                    verts_filtered = verts
                    faces_filtered = faces
                
                if len(verts_filtered) > 0 and len(faces_filtered) > 0:
                    # Plot the surface
                    ax1.plot_trisurf(verts_filtered[:, 0], verts_filtered[:, 1], verts_filtered[:, 2], 
                                    triangles=faces_filtered, alpha=0.5, color='lightcoral', 
                                    label='GT Surface')
                
        except Exception as e:
            print(f"Warning: Could not create GT surface: {e}")
            # Fallback to points
            if len(gt_samples_filtered) > 0:
                ax1.scatter(gt_samples_filtered[:, 0], gt_samples_filtered[:, 1], gt_samples_filtered[:, 2], 
                           c='red', s=20, alpha=0.6, label='Ground Truth')
    else:
        # Plot ground truth samples as points (filtered)
        if len(gt_samples_filtered) > 0:
            ax1.scatter(gt_samples_filtered[:, 0], gt_samples_filtered[:, 1], gt_samples_filtered[:, 2], 
                       c='red', s=50, alpha=0.95, label='Ground Truth', zorder=100)
    
    # Layout settings
    ax1.set_xlim([bounds[0], bounds[1]])
    ax1.set_ylim([bounds[2], bounds[3]])
    ax1.set_zlim([bounds[4], bounds[5]])
    
    # Set aspect ratio based on actual data ranges
    try:
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_range = max(x_range, y_range, z_range)
        ax1.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
    except Exception:
        pass
    
    ax1.set_xlabel('X', fontsize=16, labelpad=10)
    ax1.set_ylabel('Y', fontsize=16, labelpad=10)
    ax1.set_zlabel('Z', fontsize=16, labelpad=10)
    ax1.tick_params(labelsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=14)
    
    # Update title based on constraint
    title_suffix = "(Hemisphere y≥0)" if constraint_name in ["sphere", "ellipsoid"] else ""
    ax1.set_title(f'3D Projections {title_suffix}', fontsize=16, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    
    # Right plot: Function values before and after projection
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Update values with filtered points
    f_vals_show = f_vals_np[:n_pos_show + n_neg_show]
    proj_f_vals_show = proj_f_vals_np[:n_pos_show + n_neg_show]
    
    # Plot function values
    indices = np.arange(len(f_vals_show))
    ax2.bar(indices, np.abs(f_vals_show), alpha=0.7, label='|f(x)| Original', color='lightcoral')
    ax2.bar(indices, np.abs(proj_f_vals_show), alpha=0.7, label='|f(x\')| Projected', color='lightblue')
    
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('|Function Value|', fontsize=14)
    ax2.set_xlabel('Point Index', fontsize=14)
    ax2.set_title('Function Values: Original vs Projected', fontsize=16)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add text with statistics - handle NaN values
    f_vals_clean = f_vals_show[~np.isnan(f_vals_show)]
    proj_f_vals_clean = proj_f_vals_show[~np.isnan(proj_f_vals_show)]
    
    mean_orig = np.mean(np.abs(f_vals_clean)) if len(f_vals_clean) > 0 else 0.0
    mean_proj = np.mean(np.abs(proj_f_vals_clean)) if len(proj_f_vals_clean) > 0 else 0.0
    
    # Count NaN values
    nan_orig = np.sum(np.isnan(f_vals_show))
    nan_proj = np.sum(np.isnan(proj_f_vals_show))
    
    stats_text = f'Mean |f(x)|: {mean_orig:.4f}\nMean |f(x\')|: {mean_proj:.4f}'
    stats_text += f'\nPoints shown: pos={n_pos_show}, neg={n_neg_show}'
    if nan_orig > 0 or nan_proj > 0:
        stats_text += f'\nNaN: orig={nan_orig}, proj={nan_proj}'
    
    ax2.text(0.02, 0.98, stats_text, 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    
    plt.show()



# ============================================================================
# AUTO SLICE VISUALIZATION WITH SMART VIEW SELECTION
# ============================================================================

def _device_of(model):
    """Get device of model parameters."""
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")

def _eval_field(model, pts_np, device):
    """Evaluate model on numpy points."""
    with torch.no_grad():
        t = torch.tensor(pts_np, dtype=torch.float32, device=device)
        vals = model(t).detach().cpu().numpy().reshape(-1)
    return vals


def _pick_k_centers(P, k):
    """Soft clustering: if fewer points than k, repeat; otherwise k-means."""
    if P.shape[0] == 0:
        return np.zeros((0, 2))
    if P.shape[0] <= k:
        # if few points, use them all and duplicate furthest until k
        reps = [P[i % P.shape[0]] for i in range(k)]
        return np.vstack(reps)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(P)
    return km.cluster_centers_



def plot_3d_slices_auto(
    model,
    gt_function,                      # callable h(x,y,z) -> float
    n_views=5,                        # Now 5 columns
    bounds=(-2.5, 2.5, -2.5, 2.5, -2.5, 2.5),
    resolution=160,                   # evaluation resolution in each plane
    title="2D Slices (full domain views)",
    constraint_name="",
    save_path=None
):
    """
    Create 2D slice visualization with FULL DOMAIN views (no zoom).
    
    Layout: 3 rows × 5 columns
    - Row 1: Z constant (XY planes) - each column shows different Z value
    - Row 2: Y constant (XZ planes) - each column shows different Y value
    - Row 3: X constant (YZ planes) - each column shows different X value
    
    Args:
        model: Trained implicit model
        gt_function: Ground truth function h(x,y,z)
        n_views: Number of columns (default 5)
        bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution: Grid resolution for evaluation
        title: Title suffix
        constraint_name: Name of constraint
        save_path: Path to save figure (optional)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    device = _device_of(model)
    model.eval()

    # 3 rows × n_views columns: each row is a different axis, columns are different values
    fig, axes = plt.subplots(3, n_views, figsize=(4 * n_views, 11))
    fig.suptitle(f"2D Slices: {constraint_name} — {title}", fontsize=16, fontweight="bold")
    
    if constraint_name == "sphere":
        interesting_range = 0.9
        z_vals = np.linspace(-interesting_range, interesting_range, n_views)
        y_vals = np.linspace(-interesting_range, interesting_range, n_views)
        x_vals = np.linspace(-interesting_range, interesting_range, n_views)       
    elif constraint_name == "hyperplane":
        interesting_range = 1.9
        z_vals = np.linspace(-interesting_range, interesting_range, n_views)
        y_vals = np.linspace(-interesting_range, interesting_range, n_views)
        x_vals = np.linspace(-interesting_range, interesting_range, n_views)
    elif constraint_name == "cylinder":
        interesting_range_x_y = 0.9
        interesting_range_z = 1.9
        z_vals = np.linspace(-interesting_range_z, interesting_range_z, n_views)
        y_vals = np.linspace(-interesting_range_x_y, interesting_range_x_y, n_views)
        x_vals = np.linspace(-interesting_range_x_y, interesting_range_x_y, n_views)
    elif constraint_name == "ellipsoid":
        interesting_range_x = 1.4
        interesting_range_y_z = 0.5
        z_vals = np.linspace(-interesting_range_y_z, interesting_range_y_z, n_views)
        y_vals = np.linspace(-interesting_range_y_z, interesting_range_y_z, n_views)
        x_vals = np.linspace(-interesting_range_x, interesting_range_x, n_views)  
    else:
        interesting_range_x_y = 1.2
        interesting_range_z = 1.8
        z_vals = np.linspace(-interesting_range_z, interesting_range_z, n_views)
        y_vals = np.linspace(-interesting_range_x_y, interesting_range_x_y, n_views)
        x_vals = np.linspace(-interesting_range_x_y, interesting_range_x_y, n_views)


    # Check if spiral (1D curve) for special rendering
    # Note: spiral_tube is a 2D surface, NOT a 1D curve
    is_spiral = (constraint_name.lower() == "spiral")
    
    # ROW 1: Z constant (XY planes) - each column is different Z
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X_xy, Y_xy = np.meshgrid(x, y)
    
    for col_idx, z_const in enumerate(z_vals):
        ax = axes[0, col_idx]
        Z = np.full_like(X_xy, z_const)
        pts = np.stack([X_xy.ravel(), Y_xy.ravel(), Z.ravel()], axis=1)
        gt_vals = np.array([gt_function(p) for p in pts]).reshape(X_xy.shape)
        
        # Plot GT
        if is_spiral:
            threshold = 0.05
            mask = np.abs(gt_vals) < threshold
            gt_x = X_xy[mask]
            gt_y = Y_xy[mask]
            if len(gt_x) > 0:
                ax.scatter(gt_x, gt_y, c='red', s=15, marker='o', label='GT' if col_idx == 0 else '', 
                          zorder=10, edgecolors='darkred', linewidths=0.5, alpha=0.7)
        else:
            ax.contour(X_xy, Y_xy, gt_vals, levels=[0], colors="red", linewidths=2.2)
        
        # Plot model
        with torch.no_grad():
            mdl_vals = _eval_field(model, pts, device).reshape(X_xy.shape)
        ax.contour(X_xy, Y_xy, mdl_vals, levels=[0], colors="blue", linewidths=2.0, linestyles="--")
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Y", fontsize=9)
        ax.set_title(f"XY (Z={z_const:.2f})", fontsize=10)
        ax.tick_params(labelsize=8)
        
        # Legend on first column
        if col_idx == 0:
            from matplotlib.lines import Line2D
            if is_spiral:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, 
                          label="GT", markeredgecolor='darkred', markeredgewidth=0.5),
                    Line2D([0], [0], color="blue", lw=2.0, ls="--", label="Learned")
                ]
            else:
                legend_elements = [
                    Line2D([0], [0], color="red", lw=2.2, label="GT"),
                    Line2D([0], [0], color="blue", lw=2.0, ls="--", label="Learned")
                ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    # ROW 2: Y constant (XZ planes) - each column is different Y
    x_grid = np.linspace(x_min, x_max, resolution)
    z_grid = np.linspace(z_min, z_max, resolution)
    X_xz, Z_xz = np.meshgrid(x_grid, z_grid)
    
    for col_idx, y_const in enumerate(y_vals):
        ax = axes[1, col_idx]
        Y = np.full_like(X_xz, y_const)
        pts = np.stack([X_xz.ravel(), Y.ravel(), Z_xz.ravel()], axis=1)
        gt_vals = np.array([gt_function(p) for p in pts]).reshape(X_xz.shape)
        
        # Plot GT
        if is_spiral:
            threshold = 0.05
            mask = np.abs(gt_vals) < threshold
            gt_x = X_xz[mask]
            gt_z = Z_xz[mask]
            if len(gt_x) > 0:
                ax.scatter(gt_x, gt_z, c='red', s=15, marker='o', 
                          zorder=10, edgecolors='darkred', linewidths=0.5, alpha=0.7)
        else:
            ax.contour(X_xz, Z_xz, gt_vals, levels=[0], colors="red", linewidths=2.2)
        
        # Plot model
        with torch.no_grad():
            mdl_vals = _eval_field(model, pts, device).reshape(X_xz.shape)
        ax.contour(X_xz, Z_xz, mdl_vals, levels=[0], colors="blue", linewidths=2.0, linestyles="--")
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X", fontsize=9)
        ax.set_ylabel("Z", fontsize=9)
        ax.set_title(f"XZ (Y={y_const:.2f})", fontsize=10)
        ax.tick_params(labelsize=8)
    
    # ROW 3: X constant (YZ planes) - each column is different X
    y_grid = np.linspace(y_min, y_max, resolution)
    z_grid = np.linspace(z_min, z_max, resolution)
    Y_yz, Z_yz = np.meshgrid(y_grid, z_grid)
    
    for col_idx, x_const in enumerate(x_vals):
        ax = axes[2, col_idx]
        X = np.full_like(Y_yz, x_const)
        pts = np.stack([X.ravel(), Y_yz.ravel(), Z_yz.ravel()], axis=1)
        gt_vals = np.array([gt_function(p) for p in pts]).reshape(Y_yz.shape)
        
        # Plot GT
        if is_spiral:
            threshold = 0.05
            mask = np.abs(gt_vals) < threshold
            gt_y = Y_yz[mask]
            gt_z = Z_yz[mask]
            if len(gt_y) > 0:
                ax.scatter(gt_y, gt_z, c='red', s=15, marker='o', 
                          zorder=10, edgecolors='darkred', linewidths=0.5, alpha=0.7)
        else:
            ax.contour(Y_yz, Z_yz, gt_vals, levels=[0], colors="red", linewidths=2.2)
        
        # Plot model
        with torch.no_grad():
            mdl_vals = _eval_field(model, pts, device).reshape(Y_yz.shape)
        ax.contour(Y_yz, Z_yz, mdl_vals, levels=[0], colors="blue", linewidths=2.0, linestyles="--")
        
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Y", fontsize=9)
        ax.set_ylabel("Z", fontsize=9)
        ax.set_title(f"YZ (X={x_const:.2f})", fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved full-domain slices to {save_path}")
    plt.show()
    return fig


def plot_3d_slices_grid(results, archs, loss_variants, gt_function, 
                        bounds=[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
                        resolution=160, save_dir=None, constraint_name=""):
    """
    Generate automatic slice visualizations for multiple models.
    
    Uses intelligent view selection with zero-crossing detection and K-means clustering
    to automatically find the most interesting regions to visualize.
    
    Args:
        results: Dict mapping (arch_name, loss_name) → (model, train_curve, test_curve)
        archs: List of (arch_name, make_model, epochs, lr)
        loss_variants: List of (variant_id, variant_name)
        gt_function: Ground truth function
        bounds: Spatial bounds
        resolution: Grid resolution
        save_dir: Directory to save figures (optional)
        constraint_name: Name of constraint
        
    """
    import os
    
    for arch_name, _, _, _ in archs:
        for vid, vname in loss_variants:
            key = (arch_name, vname)
            if key not in results:
                continue
            
            model = results[key][0]
            title = f"{arch_name} - {vname}"
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"slices_{arch_name}_{vname.replace(' ', '_')}.png"
                save_path = os.path.join(save_dir, fname)
            else:
                save_path = None
            
            # Always use automatic intelligent view selection
            plot_3d_slices_auto(
                model=model,
                gt_function=gt_function,
                n_views=5,
                bounds=bounds,
                resolution=resolution,
                title=title,
                constraint_name=constraint_name,
                save_path=save_path
            )
