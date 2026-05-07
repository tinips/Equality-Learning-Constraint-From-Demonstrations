"""
Visualization utilities for implicit constraint learning models.
Contains functions that work for both 2D and 3D data.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_confusion_matrix_grid(results, archs, loss_variants, pos_test, neg_test, save_path=None, pos_eps=0.04, neg_thr=0.08):
	"""
	Plota una graella de matrius de confusió per tots els models i variants.
	Works for both 2D and 3D data.
	Args:
		results: Dict amb (arch_name, loss_name) -> (model, ...)
		archs: Llista de tuples (arch_name, ...)
		loss_variants: Llista de tuples (_, loss_name)
		pos_test: Tensor de mostres positives (test) - (N, 2) or (N, 3)
		neg_test: Tensor de mostres negatives (test) - (N, 2) or (N, 3)
		save_path: Si s'indica, desa la figura en aquest path
		pos_eps, neg_thr: Llindars per la matriu
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
			model = results[(arch_name, loss_name)][0]
			title = f"{arch_name} | {loss_name}"
			plot_confusion_matrix(model, pos_test, neg_test, ax=axes[r, c], title=title, pos_eps=pos_eps, neg_thr=neg_thr)
	fig.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=180)
		plt.close(fig)
	else:
		plt.show()


def plot_confusion_matrix(model, pos_test, neg_test, pos_eps, neg_thr, ax=None, title=None):
	"""
	Plota la matriu de confusió per un model donat mostres positives i negatives.
	Works for both 2D and 3D data.
	Args:
		model: Model entrenat
		pos_test: Tensor de mostres positives (N, 2) or (N, 3)
		neg_test: Tensor de mostres negatives (N, 2) or (N, 3)
		pos_eps: Llindar per considerar una mostra positiva com a correcta
		neg_thr: Llindar per considerar una mostra negativa com a correcta
		ax: Matplotlib axis (opcional)
		title: Títol del plot (opcional)
	"""
	with torch.no_grad():
		h_pos = model(pos_test)
		h_neg = model(neg_test)
		# Convert to numpy for safe printing and robust min/max handling
		h_pos_np = h_pos.abs().cpu().numpy()
		h_neg_np = h_neg.abs().cpu().numpy()
		print("Sortides positives (|f(x)|):", h_pos_np)
		print("Sortides negatives (|f(x)|):", h_neg_np)
		if h_pos_np.size > 0:
			print("Min/Max pos:", float(h_pos_np.min()), float(h_pos_np.max()))
		else:
			print("Min/Max pos: (empty)")
		if h_neg_np.size > 0:
			print("Min/Max neg:", float(h_neg_np.min()), float(h_neg_np.max()))
		else:
			print("Min/Max neg: (empty)")
		h_pos = h_pos.abs()
		h_neg = h_neg.abs()
		TP = (h_pos <= pos_eps).sum().item()
		FN = (h_pos >  pos_eps).sum().item()
		TN = (h_neg >= neg_thr).sum().item()
		FP = (h_neg <  neg_thr).sum().item()
		n_pos = pos_test.shape[0]
		n_neg = neg_test.shape[0]
	cm = np.array([[TP, FN], [FP, TN]], dtype=int)
	cm_pct = np.zeros_like(cm, dtype=float)
	cm_pct[0,0] = TP / n_pos * 100 if n_pos > 0 else 0
	cm_pct[0,1] = FN / n_pos * 100 if n_pos > 0 else 0
	cm_pct[1,0] = FP / n_neg * 100 if n_neg > 0 else 0
	cm_pct[1,1] = TN / n_neg * 100 if n_neg > 0 else 0
	if ax is None:
		fig, ax = plt.subplots(figsize=(4,4))
		show_plot = True
	else:
		show_plot = False
	im = ax.imshow(cm, cmap="Blues")
	for i in range(2):
		for j in range(2):
			txt = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"
			ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=14)
	ax.set_xticks([0,1]); ax.set_yticks([0,1])
	ax.set_xticklabels(["Pos", "Neg"])
	ax.set_yticklabels(["Pred Pos", "Pred Neg"])
	ax.set_xlabel("True label"); ax.set_ylabel("Prediction")
	if title:
		ax.set_title(title)
	else:
		ax.set_title("Confusion Matrix")
	plt.tight_layout()
	if show_plot:
		plt.show()
	return ax


def plot_curve_grid(results, archs, loss_variants, save_path=None):
	"""
	Plots a grid of training curves for all architectures and loss variants.
	Works for both 2D and 3D data.
	Args:
		results: Dict with (arch_name, loss_name) -> (model, curve)
		archs: List of (arch_name, ...)
		loss_variants: List of (id, loss_name)
		save_path: If provided, saves the figure to this path
	"""
	rows = len(archs)
	cols = len(loss_variants)
	fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4.5*rows))
	if rows == 1:
		axes = np.expand_dims(axes, 0)
	if cols == 1:
		axes = np.expand_dims(axes, 1)
	for r, (arch_name, *_ ) in enumerate(archs):
		for c, (_, loss_name) in enumerate(loss_variants):
			ax = axes[r, c]
			res = results[(arch_name, loss_name)]
			if len(res) == 2:
				_, curve = res
				ax.plot(curve, color='royalblue', linewidth=2.5, label="Train Loss")
				ax.set_title(f"{arch_name} | {loss_name}", fontsize=15)
				ax.set_xlabel("Epoch", fontsize=12)
				ax.set_ylabel("Loss", fontsize=12)
				ax.grid(True, alpha=0.4, linestyle='--')
				ax.legend(fontsize=11)
			elif len(res) == 3:
				_, train_curve, test_curve = res
				epochs = np.arange(len(train_curve))
				ax.plot(epochs, train_curve, color='royalblue', linewidth=2.5, label="Train Loss")
				ax.plot(epochs, test_curve, color='crimson', linewidth=2.5, label="Test Loss")
				# Punt mínim test loss
				min_test_idx = np.argmin(test_curve)
				min_test_epoch = epochs[min_test_idx]
				min_test_loss = test_curve[min_test_idx]
				ax.scatter([min_test_epoch], [min_test_loss], color='crimson', s=80, zorder=5, label=f"Min Test Loss ({min_test_epoch})")
				ax.set_title(f"{arch_name} | {loss_name}", fontsize=15)
				ax.set_xlabel("Epoch", fontsize=12)
				ax.set_ylabel("Loss", fontsize=12)
				ax.grid(True, alpha=0.4, linestyle='--')
				ax.legend(fontsize=11, loc='upper right')
				# Anotació amb valor mínim
				ax.text(0.98, 0.02, f"Final Train: {train_curve[-1]:.4f}\nFinal Test: {test_curve[-1]:.4f}",
						transform=ax.transAxes, fontsize=10, ha='right', va='bottom', 
						bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
	fig.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=220)
		plt.close(fig)
	else:
		plt.show()


def plot_learning_curves(results, save_path=None):
	"""
	Plota corbes d'aprenentatge (train/test loss vs epochs) amb marcatge del mínim test loss.
	Works for both 2D and 3D data.
	Args:
		results: Dict amb (arch_name, loss_name) -> (model, curve_data)
		save_path: Si s'indica, desa la figura en aquest path
	"""
	n = len(results)
	fig, axes = plt.subplots(1, n, figsize=(8*n, 6))
	if n == 1:
		axes = [axes]
	for idx, ((arch_name, loss_name), (model, curve_data)) in enumerate(results.items()):
		ax = axes[idx]
		train_losses = curve_data['train_losses']
		test_losses = curve_data['test_losses']
		eval_epochs = curve_data['eval_epochs']
		ax.plot(eval_epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
		ax.plot(eval_epochs, test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
		ax.set_xlabel('Epochs', fontsize=12)
		ax.set_ylabel('Loss', fontsize=12)
		ax.set_title(f'{arch_name} | {loss_name}\nLearning Curves', fontsize=14)
		ax.legend(fontsize=10)
		ax.grid(True, alpha=0.3)
		min_test_idx = np.argmin(test_losses)
		min_test_epoch = eval_epochs[min_test_idx]
		min_test_loss = test_losses[min_test_idx]
		ax.scatter([min_test_epoch], [min_test_loss], color='red', s=100, zorder=5,
				   label=f'Best Test (epoch {min_test_epoch})')
		final_train = train_losses[-1]
		final_test = test_losses[-1]
		ax.text(0.98, 0.02, f'Final Train: {final_train:.4f}\nFinal Test: {final_test:.4f}',
				transform=ax.transAxes, fontsize=10, ha='right', va='bottom', 
				bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
		plt.close()
	else:
		plt.show()

		

from src.data.datasets import make_constraint
from src.data.datasets3D import make_constraint_3d
def plot_heatmap(
    archs,
    loss_variants,
    name,
    results,
    pos_tensor,
    config,
    save_path=None,
    neg_tensor=None,
	is_3d=False
):
	"""
	Plots a heatmap of model performance.
	- Si es passa neg_tensor: mostra accuracy (ACC) sobre el test set (train/test split).
	- Si NO es passa neg_tensor: mostra DS/GS/finalscore (mètrica geomètrica clàssica).

	Args:
		archs: List of (arch_name, ...)
		loss_variants: List of (id, loss_name)
		name: Constraint name (str)
		results: Dict with (arch_name, loss_name) -> (model, ...)
		pos_tensor: Positive samples (Tensor, test set si ACC)
		config: Config object with TAU_DIST, SIGMA_GEOM, ALPHA_FINAL
		grid_zero_contour_points_fn: Function to extract model zero-level set points
		save_path: If provided, saves the figure to this path
		neg_tensor: (opcional) Negatives (Tensor, test set). Si es passa, es mostra ACC.
	"""
	arch_names = [a[0] for a in archs]
	loss_names = [v[1] for v in loss_variants]

	data_sat_mat   = np.zeros((len(archs), len(loss_variants)))
	geom_score_mat = np.zeros((len(archs), len(loss_variants)))
	final_mat      = np.zeros((len(archs), len(loss_variants)))
	chamfer_mat    = np.zeros_like(final_mat)
	accuracy_mat   = np.zeros_like(final_mat)

	if is_3d:
		_, _, _, gt_curve_fn = make_constraint_3d(name)
	else:
		_, _, _, gt_curve_fn = make_constraint(name)

	gt_dense = gt_curve_fn(M=1200)
	from src.utils.metrics import data_sat_score, geometry_chamfer_score

	# Decideix automàticament: ACC si neg_tensor, DS/GS si no
	use_acc = neg_tensor is not None
	for r, arch in enumerate(arch_names):
		for c, loss in enumerate(loss_names):
			res = results[(arch, loss)]
			model = res[0]
			if use_acc:
				from src.utils.eval import eval_accuracy_simple
				acc = eval_accuracy_simple(model, pos_tensor, neg_tensor)
				accuracy_mat[r, c] = acc['global_accuracy']  # Extrae el valor float correcte
			else:
				ds, _ = data_sat_score(model, pos_tensor, tau=config.TAU_DIST)
				gs, cd = geometry_chamfer_score(
					model, 
					gt_dense, 
					sigma=config.SIGMA_GEOM, 
				)
				data_sat_mat[r, c]   = ds
				geom_score_mat[r, c] = gs
				chamfer_mat[r, c]    = cd
				final_mat[r, c]      = (ds**config.ALPHA_FINAL) * (gs**(1-config.ALPHA_FINAL))

	plt.figure(figsize=(8.2,6.2))
	if use_acc:
		im = plt.imshow(accuracy_mat, vmin=0.0, vmax=1.0, cmap="viridis")
		for i in range(accuracy_mat.shape[0]):
			for j in range(accuracy_mat.shape[1]):
				txt = f"{accuracy_mat[i,j]:.2f}"
				plt.text(j, i, txt, ha='center', va='center', color='w', fontsize=10)
		plt.colorbar(im, label="Accuracy")
		plt.title(f"Accuracy heatmap  |  GT: {name}")
	else:
		im = plt.imshow(final_mat, vmin=0.0, vmax=1.0, cmap="viridis")
		for i in range(final_mat.shape[0]):
			for j in range(final_mat.shape[1]):
				txt = f"{final_mat[i,j]:.2f}\nDS:{data_sat_mat[i,j]:.2f}\nGS:{geom_score_mat[i,j]:.2f}"
				plt.text(j, i, txt, ha='center', va='center', color='w', fontsize=8)
		plt.colorbar(im, label="FinalScore")
		plt.title(f"Scale-invariant accuracy heatmap  |  GT: {name}")
	plt.xticks(range(len(loss_names)), loss_names, rotation=20)
	plt.yticks(range(len(arch_names)), arch_names)
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=200, bbox_inches='tight')
	else:
		plt.show()



def plot_comparasion(results, save_path=None):
	plt.figure(figsize=(10, 6))
	tab_colors = plt.get_cmap('tab10')
	for idx, ((n_pos, n_neg), curve) in enumerate(results.items()):
		if not isinstance(curve, dict):
			print(f"[ERROR] Entrada a results_curve per ({n_pos}, {n_neg}) no és un dict: {type(curve)}. Valor: {curve}")
			continue
		epochs = curve.get('eval_epochs', None)
		test_loss = curve.get('test_losses', None)
		if epochs is None or test_loss is None:
			print(f"[ERROR] Entrada a results_curve per ({n_pos}, {n_neg}) no té 'eval_epochs' o 'test_losses'.")
			continue
		color = tab_colors(idx % 10)  # tab10 té 10 colors ben diferenciats
		plt.plot(epochs, test_loss, label=f"{n_pos} mostres", color=color)
	plt.xlabel('Epoch')
	plt.ylabel('Test Loss')
	plt.title('Evolution of Test Loss vs Number of Samples')
	plt.legend(title="Number of Samples")
	plt.grid(True)
	plt.tight_layout()
	if save_path is not None:
		plt.savefig(save_path)
		print(f"Figure saved as {save_path}")
	else:
		plt.savefig("test_loss_curves_vs_n_samples.png")
		print("Figure saved as test_loss_curves_vs_n_samples.png")

