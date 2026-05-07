import torch
import torch.nn.functional as F
import torch.optim as optim
from src.losses.loss_factory import pos_loss, pos_neg_loss,  knn_weighted_neg_loss, neural_pull_loss
from src.losses.regularizers import param_l2, eikonal_penalty

def compute_loss(model, xb_p, xb_n, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal=1e-3, return_projection_info=False, is_2d=False, regularizer=None):
	"""
	Compute loss with optional regularization.
	
	Args:
		regularizer: Type of regularization to apply ('none', 'l2', 'eikonal', None)
	"""
	if variant == 1:
		return pos_loss(model, xb_p)
	elif variant == 2:
		return pos_neg_loss(model, xb_p, xb_n, margin, alpha)
	
	
	elif variant == 5:
		return pos_loss(model, xb_p) + knn_weighted_neg_loss(model, xb_n, xb_p, margin, alpha)
	elif variant == 6:
		# Neural Pull Loss with optional regularization
		base_loss = neural_pull_loss(model, xb_n, xb_p, return_projection_info=return_projection_info)
		
		if return_projection_info:
			base_loss, proj_info = base_loss
			
			# Apply regularization if specified
			if regularizer == 'l2':
				reg_term = lam_l2 * param_l2(model)
				return base_loss + reg_term, proj_info
			elif regularizer == 'eikonal':
				all_points = torch.cat([xb_p, xb_n], dim=0)
				reg_term = lam_eikonal * eikonal_penalty(model, all_points, target_norm=1.0)
				return base_loss + reg_term, proj_info
			else:  # 'none' or None
				return base_loss, proj_info
		else:
			# Apply regularization if specified
			if regularizer == 'l2':
				reg_term = lam_l2 * param_l2(model)
				return base_loss + reg_term
			elif regularizer == 'eikonal':
				all_points = torch.cat([xb_p, xb_n], dim=0)
				reg_term = lam_eikonal * eikonal_penalty(model, all_points, target_norm=1.0)
				return base_loss + reg_term
			else:  # 'none' or None
				return base_loss
	
	else:
		raise ValueError(f"Unknown loss variant: {variant}. Supported variants: 1 (pos), 2 (pos+neg), 5 (knn-neg), 6 (neural-pull)")


def train_one(model, pos, neg, variant=1, epochs=300, lr=1e-3,
			  batch=128, margin=0.65, alpha=0.5, lam_l2=1e-4, lam_grad=1e-3, lam_eikonal=1e-3):
	"""
	Standard training loop for a single train set (no split).
	Args:
		model: Model to train
		pos: Positive samples (Tensor)
		neg: Negative samples (Tensor)
		variant: Loss variant (1=POS, 2=POS+NEG, 3=+L2-deprecated, 4=+GRAD, 7=Eikonal-RECOMMENDED)
		epochs: Number of epochs
		lr: Learning rate
		batch: Batch size
		margin, alpha, lam_l2, lam_grad, lam_eikonal: Loss hyperparameters
			lam_grad: weight for gradient penalty (variant 4)
			lam_eikonal: weight for Eikonal regularization (variant 7)
	Returns:
		model, losses (np.array)
	"""
	opt = optim.Adam(model.parameters(), lr=lr)
	losses = []
	Np, Nn = pos.shape[0], neg.shape[0]
	for ep in range(epochs):
		idx_p = torch.randint(0, Np, (batch,))
		xb_p = pos[idx_p]
		xb_n = None
		if variant in [2, 3, 4, 5, 6, 7]:  
			idx_n = torch.randint(0, Nn, (batch,))
			xb_n = neg[idx_n]
		
		# Use unified loss computation
		loss = compute_loss(model, xb_p, xb_n, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal)
		
		opt.zero_grad(); loss.backward(); opt.step()
		losses.append(loss.item())
	return model, losses

def train_one_with_split(model, pos_train, neg_train, pos_eval, neg_eval,
						epochs, variant=1, lr=1e-3, batch=128, margin=0.65,
						alpha=0.5, lam_l2=1e-4, lam_grad=1e-3, lam_eikonal=1e-3,
						noise_level=0.0, is_2d=False, regularizer=None):
	"""
	Training loop using train/test split, tracking both train and test losses.
	Trains for all epochs and keeps the model with best test loss.
	Args:
		model: Model to train
		pos_train, neg_train: Training samples (Tensors)
		pos_eval, neg_eval: Evaluation samples (Tensors)
		variant: Loss variant (1=POS, 2=POS+NEG, 5=KNN-NEG, 6=Neural-Pull)
		epochs: Number of epochs to train
		lr: Learning rate
		batch: Batch size
		margin, alpha, lam_l2, lam_grad, lam_eikonal: Loss hyperparameters
		regularizer: Type of regularization ('none', 'l2', 'eikonal', None)
	Returns:
		model, train_losses (list), test_losses (list), [projection_info if variant==6]
	"""


	opt = optim.Adam(model.parameters(), lr=lr)
	
	# Track best model based on test loss
	best_test_loss = float('inf')
	best_epoch = 0
	best_model_state = None
	opt = optim.Adam(model.parameters(), lr=lr)
	
	# Early stopping variables
	best_test_loss = float('inf')
	patience_counter = 0
	best_model_state = None
	train_losses = []
	test_losses = []
	Np_train, Nn_train = pos_train.shape[0], neg_train.shape[0]
	projected_info = None  
	
	for ep in range(epochs):
		idx_p = torch.randint(0, Np_train, (batch,))
		xb_p = pos_train[idx_p]
		xb_n = None
		if variant in [2, 3, 4, 5, 6, 7]: 
			idx_n = torch.randint(0, Nn_train, (batch,))
			xb_n = neg_train[idx_n]
		
		if variant == 6:
			get_proj_info = (ep == epochs - 1)
			result = compute_loss(model, xb_p, xb_n, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal, return_projection_info=get_proj_info, is_2d=is_2d, regularizer=regularizer)
			if get_proj_info:
				loss, projected_info = result
			else:
				loss = result
		else:
			loss = compute_loss(model, xb_p, xb_n, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal, regularizer=regularizer)

		opt.zero_grad(); loss.backward(); opt.step()
		train_losses.append(loss.item())

		# Neural Pull Loss (variant 6) ALWAYS needs gradients for projection computation
		# Even without eikonal regularization, the loss itself needs gradients
		if variant == 6:
			# Ensure eval tensors have requires_grad for neural pull computation
			pos_eval_grad = pos_eval.detach().requires_grad_(True)
			neg_eval_grad = neg_eval.detach().requires_grad_(True)
			test_loss = compute_loss(model, pos_eval_grad, neg_eval_grad, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal, regularizer=regularizer)
		else:
			with torch.no_grad():
				test_loss = compute_loss(model, pos_eval, neg_eval, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal, regularizer=regularizer)
		test_losses.append(test_loss.item())
		
		# Track best model based on test loss
		if test_loss.item() < best_test_loss:
			best_test_loss = test_loss.item()
			best_epoch = ep
			best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
	
	# Restore best model (from epoch with lowest test loss)
	if best_model_state is not None:
		model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in best_model_state.items()})
		print(f"    Best model: epoch {best_epoch+1}/{epochs} (test loss: {best_test_loss:.6f})")
	
	if variant == 6:
		# Use fixed indices for reproducible visualization
		torch.manual_seed(42)  # Fixed seed for reproducible batch selection
		batch_viz = min(batch, Np_train, Nn_train)
		idx_p_viz = torch.randint(0, Np_train, (batch_viz,))
		idx_n_viz = torch.randint(0, Nn_train, (batch_viz,))
		xb_p_viz = pos_train[idx_p_viz]
		xb_n_viz = neg_train[idx_n_viz]
		
		_, projected_info = compute_loss(model, xb_p_viz, xb_n_viz, variant, margin, alpha, lam_l2, lam_grad, lam_eikonal, return_projection_info=True, regularizer=regularizer)
		return model, train_losses, test_losses, projected_info		
	else:
		return model, train_losses, test_losses
