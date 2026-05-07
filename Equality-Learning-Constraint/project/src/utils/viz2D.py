"""
2D-specific visualization utilities for implicit constraint learning models.
"""

from .viz import *  


def plot_samples(pos_np, neg_np, gt_curve_fn, grid_lim, save_path=None, constraint_name=None):
	"""
	Plots positive and negative samples and overlays the ground truth curve (2D version).
	Args:
		pos_np: Positive samples (N,2)
		neg_np: Negative samples (N,2)
		gt_curve_fn: Function to generate GT curve points
		grid_lim: Plotting range
		save_path: If provided, saves the figure to this path
		constraint_name: Name of the constraint (for legend)
	"""
	plt.figure(figsize=(6,6))
	plt.scatter(neg_np[:,0], neg_np[:,1], s=4, alpha=0.3, c='orange', label="negative (normal offsets)")
	plt.scatter(pos_np[:,0], pos_np[:,1], s=12, alpha=0.8, c='blue', label="positive (on GT)")
	gt = gt_curve_fn(M=1000)
	if gt.shape[0] >= 2:
		diffs = np.linalg.norm(np.diff(gt, axis=0), axis=1)
		thr = grid_lim * 0.5
		cut_idx = np.where(diffs > thr)[0]
		starts = np.concatenate(([0], cut_idx + 1))
		ends   = np.concatenate((cut_idx + 1, [gt.shape[0]]))
		for s, e in zip(starts, ends):
			seg = gt[s:e]
			if seg.shape[0] >= 2:
				plt.plot(seg[:,0], seg[:,1], 'k-', lw=2, label=f"GT: {constraint_name}" if s==starts[0] and constraint_name else None)
	plt.axis('equal')
	plt.legend(loc='upper right')
	plt.title("Positive & Negative samples")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.grid(True, alpha=0.3)
	plt.xlim([-grid_lim, grid_lim])
	plt.ylim([-grid_lim, grid_lim])
	if save_path:
		plt.savefig(save_path, dpi=180)
		plt.close()
	else:
		plt.show()


def plot_levelset_grid(results, archs, loss_variants, gt_curve_fn, grid_lim, grid_n, save_path=None, constraint_name=None):
	"""
	Plots a grid of levelsets for all architectures and loss variants (2D version).
	Args:
		results: Dict with (arch_name, loss_name) -> (model, curve)
		archs: List of (arch_name, ...)
		loss_variants: List of (id, loss_name)
		gt_curve_fn: Function to generate GT curve points
		grid_lim: Plotting range
		grid_n: Grid resolution
		save_path: If provided, saves the figure to this path
		constraint_name: Name of the constraint (for title)
	"""
	rows = len(archs)
	cols = len(loss_variants)
	fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
	if rows == 1:
		axes = np.expand_dims(axes, 0)
	if cols == 1:
		axes = np.expand_dims(axes, 1)
	for r, (arch_name, *_ ) in enumerate(archs):
		for c, (_, loss_name) in enumerate(loss_variants):
			ax = axes[r, c]
			res = results[(arch_name, loss_name)]
			model = res[0]
			plot_levelset(ax, model, gt_curve_fn, grid_lim=grid_lim, grid_n=grid_n, title=f"{arch_name} | {loss_name} | {constraint_name}")
	fig.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=200)
		plt.close(fig)
	else:
		plt.show()


def plot_levelset(ax, model, gt_curve_fn, grid_lim=1.8, grid_n=240, title="", overlay_gt=True):
    """
    Plot the zero-level set of a model and optionally overlay the ground truth curve (2D version).
    Args:
        ax: Matplotlib axis
        model: Trained model
        gt_curve_fn: Function to generate GT curve points
        grid_lim: Plotting range
        grid_n: Grid resolution
        title: Plot title
        overlay_gt: Whether to plot GT curve
    """
    xs = np.linspace(-grid_lim, grid_lim, grid_n)
    ys = np.linspace(-grid_lim, grid_lim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        ZZ = model(torch.from_numpy(P)).cpu().numpy().reshape(grid_n, grid_n)
    max_abs = np.max(np.abs(ZZ))
    
    n_contours = 20  
    levels = np.linspace(-max_abs, max_abs, n_contours)
    
    extra_close = np.array([-0.5, -0.3, -0.1, -0.05, 0.05, 0.1, 0.3, 0.5])
    extra_far = np.array([-10.0, -5.0, -2.0, 2.0, 5.0, 10.0])
    ultra_far = np.array([-50.0, -20.0, 20.0, 50.0]) if max_abs > 20.0 else np.array([])
    
    extra_close = extra_close[(extra_close >= -max_abs) & (extra_close <= max_abs)]
    extra_far = extra_far[(extra_far >= -max_abs) & (extra_far <= max_abs)]
    ultra_far = ultra_far[(ultra_far >= -max_abs) & (ultra_far <= max_abs)]
    
    all_levels = np.concatenate([levels, extra_close, extra_far, ultra_far])
    all_levels = np.sort(np.unique(all_levels)) 
    
    ax.contour(XX, YY, ZZ, levels=all_levels, colors='gray', linewidths=1.2, alpha=0.8) 
    ax.contour(XX, YY, ZZ, levels=[0.0], colors='red', linewidths=4, alpha=1.0)  
    if overlay_gt:
        gt = gt_curve_fn(M=1000)
        if gt.shape[0] >= 2:
            diffs = np.linalg.norm(np.diff(gt, axis=0), axis=1)
            thr = grid_lim * 0.5
            cut_idx = np.where(diffs > thr)[0]
            starts = np.concatenate(([0], cut_idx + 1))
            ends   = np.concatenate((cut_idx + 1, [gt.shape[0]]))
            for s, e in zip(starts, ends):
                seg = gt[s:e]
                if seg.shape[0] >= 2:
                    ax.plot(seg[:,0], seg[:,1], 'k--', lw=1)
    ax.set_aspect('equal'); ax.set_xlim([-grid_lim, grid_lim]); ax.set_ylim([-grid_lim, grid_lim])
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')


def plot_levelset_with_samples(results, archs, loss_variants, gt_curve_fn, pos_np, neg_np, 
                              grid_lim=2.5, grid_n=240, constraint_name=None, save_path=None):
    """
    Plots a GRID of levelsets with samples for all architectures and loss variants (2D version).
    Args:
        results: Dict with (arch_name, loss_name) -> (model, curve)
        archs: List of (arch_name, ...)
        loss_variants: List of (id, loss_name)
        gt_curve_fn: Function to generate GT curve points
        pos_np: Positive samples (N,2)
        neg_np: Negative samples (N,2)
        grid_lim: Plotting range
        grid_n: Grid resolution
        constraint_name: Name of the constraint (for title)
        save_path: If provided, saves the figure to this path
    """
    rows = len(archs)
    cols = len(loss_variants)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)

    n_neg_show = min(500, len(neg_np))
    n_pos_show = min(200, len(pos_np))
    neg_idx = np.random.choice(len(neg_np), n_neg_show, replace=False)
    pos_idx = np.random.choice(len(pos_np), n_pos_show, replace=False)
    neg_show = neg_np[neg_idx]
    pos_show = pos_np[pos_idx]

    xs = np.linspace(-grid_lim, grid_lim, grid_n)
    ys = np.linspace(-grid_lim, grid_lim, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

    for r, (arch_name, *_) in enumerate(archs):
        for c, (vid, vname) in enumerate(loss_variants):
            ax = axes[r, c]
            model = results[(arch_name, vname)][0]
            
            with torch.no_grad():
                ZZ = model(torch.from_numpy(P)).cpu().numpy().reshape(grid_n, grid_n)
            
            max_abs = np.max(np.abs(ZZ))
            
            very_close_levels = np.array([-0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2])
            close_levels = np.array([-0.5, -0.3, 0.3, 0.5])
            far_levels = np.array([-10.0, -8.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.8, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0])
            very_far_levels = np.array([-50.0, -30.0, -20.0, -15.0, 15.0, 20.0, 30.0, 50.0]) if max_abs > 10.0 else np.array([])
            ultra_far_levels = np.array([-100.0, -80.0, 80.0, 100.0]) if max_abs > 50.0 else np.array([])
            
            all_levels = np.concatenate([ultra_far_levels, very_far_levels, far_levels, close_levels, very_close_levels])
            valid_levels = all_levels[(all_levels >= -max_abs) & (all_levels <= max_abs)]
            valid_levels = np.sort(valid_levels)
            
            ax.contour(XX, YY, ZZ, levels=valid_levels, colors='gray', linewidths=1.2, alpha=0.8)
            
            ax.contour(XX, YY, ZZ, levels=[0.0], colors='red', linewidths=4, alpha=1.0)
            
            ax.scatter(neg_show[:,0], neg_show[:,1], s=8, alpha=0.4, c='orange', label="Negative", zorder=3)
            ax.scatter(pos_show[:,0], pos_show[:,1], s=15, alpha=0.8, c='blue', label="Positive", zorder=4)
            
            gt = gt_curve_fn(M=1000)
            if gt.shape[0] >= 2:
                diffs = np.linalg.norm(np.diff(gt, axis=0), axis=1)
                thr = grid_lim * 0.5
                cut_idx = np.where(diffs > thr)[0]
                starts = np.concatenate(([0], cut_idx + 1))
                ends = np.concatenate((cut_idx + 1, [gt.shape[0]]))
                
                for i, (s, e) in enumerate(zip(starts, ends)):
                    seg = gt[s:e]
                    if seg.shape[0] >= 2:
                        label_gt = "Ground Truth" if i == 0 else None
                        ax.plot(seg[:,0], seg[:,1], 'k--', lw=2, label=label_gt, alpha=0.8, zorder=5)
            
            ax.set_xlim([-grid_lim, grid_lim])
            ax.set_ylim([-grid_lim, grid_lim])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            title_color = 'blue' if "NPL" in vname else 'red'
            ax.set_title(f'{arch_name}\n{vname}', 
                        fontsize=12, fontweight='bold', color=title_color)
            
            if r == 0 and c == 0:
                ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_projection_visualization(projection_info, gt_curve_fn, grid_lim=2.5, save_path=None, n_show=50):
    """
    Visualize projection pairs (x, x') using pre-computed projection information (2D version).
    This is more efficient than recomputing projections.
    
    Args:
        projection_info: Dictionary with projection data from neural_pull_loss
        gt_curve_fn: Ground truth curve function
        grid_lim: Plot limits
        save_path: If provided, saves the figure
        n_show: Number of points to show for each type
    """
    original_np = projection_info['original_points'].numpy()
    projected_np = projection_info['projected_points'].numpy()
    f_vals_np = projection_info['original_f_vals'].numpy()
    proj_f_vals_np = projection_info['projected_f_vals'].numpy()
    n_pos = projection_info['n_pos']
    n_neg = projection_info['n_neg']
    
    n_pos_show = min(n_show, n_pos)
    n_neg_show = min(n_show, n_neg)
    
    pos_orig = original_np[:n_pos_show]
    pos_proj = projected_np[:n_pos_show]
    neg_orig = original_np[n_pos:n_pos+n_neg_show] if n_neg > 0 else np.empty((0, 2))
    neg_proj = projected_np[n_pos:n_pos+n_neg_show] if n_neg > 0 else np.empty((0, 2))
    
    plt.figure(figsize=(12, 5))
    
    # Left plot: Original vs Projected points
    plt.subplot(1, 2, 1)
    
    # Plot ground truth curve
    gt_curve = gt_curve_fn(M=800)
    if len(gt_curve) > 0:
        plt.plot(gt_curve[:, 0], gt_curve[:, 1], 'k-', linewidth=2, alpha=0.8, label='Ground Truth')
    
    # Plot positive points and projections
    if len(pos_orig) > 0:
        plt.scatter(pos_orig[:, 0], pos_orig[:, 1], c='blue', s=30, alpha=0.7, label='Pos Original')
        plt.scatter(pos_proj[:, 0], pos_proj[:, 1], c='lightblue', s=20, alpha=0.8, label='Pos Projected')
    
    # Plot negative points and projections  
    if len(neg_orig) > 0:
        plt.scatter(neg_orig[:, 0], neg_orig[:, 1], c='red', s=30, alpha=0.7, label='Neg Original')
        plt.scatter(neg_proj[:, 0], neg_proj[:, 1], c='pink', s=20, alpha=0.8, label='Neg Projected')
    
    # Draw projection arrows for a subset
    n_arrows = min(50, n_pos_show)
    for i in range(n_arrows):
        if i < len(pos_orig):
            plt.arrow(pos_orig[i, 0], pos_orig[i, 1], 
                     pos_proj[i, 0] - pos_orig[i, 0], pos_proj[i, 1] - pos_orig[i, 1],
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.5)
    
    n_arrows_neg = min(50, n_neg_show)
    for i in range(n_arrows_neg):
        if i < len(neg_orig):
            plt.arrow(neg_orig[i, 0], neg_orig[i, 1],
                     neg_proj[i, 0] - neg_orig[i, 0], neg_proj[i, 1] - neg_orig[i, 1], 
                     head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.5)
    
    plt.xlim([-grid_lim, grid_lim])
    plt.ylim([-grid_lim, grid_lim])
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('2D Original Points vs Projections')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Right plot: Function values before and after projection
    plt.subplot(1, 2, 2)
    
    # Limit to shown points
    f_vals_show = f_vals_np[:n_pos_show + n_neg_show]
    proj_f_vals_show = proj_f_vals_np[:n_pos_show + n_neg_show]
    
    # Plot function values
    indices = np.arange(len(f_vals_show))
    plt.bar(indices, np.abs(f_vals_show), alpha=0.7, label='|f(x)| Original', color='lightcoral')
    plt.bar(indices, np.abs(proj_f_vals_show), alpha=0.7, label='|f(x\')| Projected', color='lightblue')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('|Function Value|')
    plt.xlabel('Point Index')
    plt.title('2D Function Values: Original vs Projected')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text with statistics - handle NaN values
    f_vals_clean = f_vals_show[~np.isnan(f_vals_show)]
    proj_f_vals_clean = proj_f_vals_show[~np.isnan(proj_f_vals_show)]
    
    mean_orig = np.mean(np.abs(f_vals_clean)) if len(f_vals_clean) > 0 else 0.0
    mean_proj = np.mean(np.abs(proj_f_vals_clean)) if len(proj_f_vals_clean) > 0 else 0.0
    
    # Count NaN values
    nan_orig = np.sum(np.isnan(f_vals_show))
    nan_proj = np.sum(np.isnan(proj_f_vals_show))
    
    stats_text = f'Mean |f(x)|: {mean_orig:.4f}\nMean |f(x\')|: {mean_proj:.4f}'
    if nan_orig > 0 or nan_proj > 0:
        stats_text += f'\nNaN: orig={nan_orig}, proj={nan_proj}'
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
