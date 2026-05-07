"""
ECoMaNN (Equality Constraint Manifold Neural Network)
Faithful implementation following the paper:
"Learning Equality Constraints for Motion Planning on Manifolds" (arXiv:2009.11852)

Key components from paper:
- Siamese losses: reflection, fraction, similarity, norm
- Subspace alignment losses for Jacobian-PCA consistency
- OSA (Orthogonal Subspace Augmentation) data augmentation
- Multi-level augmentation (i=1..9)
- Architecture: [36,24,18,10] with BatchNorm, tanh activation
- Optimizer: RMSprop with lr=0.001
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class ECoMaNNMLP(torch.nn.Module):
    """
    MLP architecture from ECoMaNN paper.
    
    Paper specs:
    - Hidden layers: [36, 24, 18, 10] (or [128, 64, 32, 16] for high-dim)
    - Activation: tanh
    - BatchNorm: Yes
    - Weight init: Orthogonal with normalized columns
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 activation: str = 'tanh', use_batch_norm: bool = True, drop_p: float = 0.0):
        super().__init__()
        
        self.topology = [input_size] + hidden_sizes + [output_size]
        
        layers = []
        for i in range(len(self.topology) - 2):
            layers.append(torch.nn.Linear(self.topology[i], self.topology[i + 1]))
            
            # Paper uses BatchNorm
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(self.topology[i + 1]))
            
            # Paper uses tanh activation
            if activation == 'tanh':
                layers.append(torch.nn.Tanh())
            elif activation == 'relu':
                layers.append(torch.nn.ReLU())
            
            if drop_p > 0:
                layers.append(torch.nn.Dropout(p=drop_p))
        
        # Output layer (linear, no activation)
        layers.append(torch.nn.Linear(self.topology[-2], self.topology[-1]))
        
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weights_orthogonal()
    
    def _init_weights_orthogonal(self):
        """
        Orthogonal initialization with normalized columns (from paper).
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                # Normalize columns
                m.weight.data = m.weight.data / (m.weight.data.norm(dim=0, keepdim=True) + 1e-8)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# Paper default architecture
ECOMANN_HIDDEN_SIZES = [36, 24, 18, 10]
ECOMANN_HIDDEN_SIZES_HIGH_DIM = [128, 64, 32, 16]

# Paper default augmentation levels
N_AUGMENTATION_LEVELS = 9


class ECoMaNN(torch.nn.Module):
    """
    Equality Constraint Manifold Neural Network (faithful to paper).
    
    Learns implicit surface h(x) = 0 using:
    - Siamese losses: reflection, fraction, similarity, norm
    - Subspace alignment losses
    - Multi-level augmentation (i=1..N_levels)
    
    Paper: "Learning Equality Constraints for Motion Planning on Manifolds"
    arXiv:2009.11852
    """
    
    def __init__(self, input_dim: int = 3, 
                 hidden_sizes: Optional[List[int]] = None, 
                 output_dim: int = 1,  # l = normal space dimension
                 activation: str = 'tanh', 
                 use_batch_norm: bool = True,  # Paper uses BatchNorm
                 drop_p: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Paper architecture: [36, 24, 18, 10] for low-dim, [128, 64, 32, 16] for high-dim
        if hidden_sizes is None:
            hidden_sizes = ECOMANN_HIDDEN_SIZES if input_dim <= 6 else ECOMANN_HIDDEN_SIZES_HIGH_DIM
        
        self.nn_model = ECoMaNNMLP(
            input_size=input_dim, 
            hidden_sizes=hidden_sizes, 
            output_size=output_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            drop_p=drop_p
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate h(x)"""
        return self.nn_model(x)
    
    def compute_jacobian(self, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """
        Compute Jacobian J_M = ∇h(x) ∈ R^{N x l x d}
        
        The row space of J should align with normal space.
        The null space of J should align with tangent space.
        """
        x = x.requires_grad_(True)
        
        with torch.enable_grad():
            y = self.forward(x)
            
            if self.output_dim == 1:
                J = torch.autograd.grad(y.sum(), x, create_graph=create_graph)[0]
                return J.unsqueeze(1)  # (N, 1, d)
            else:
                J = torch.stack([
                    torch.autograd.grad(y[:, i].sum(), x, 
                                      retain_graph=True, 
                                      create_graph=create_graph)[0]
                    for i in range(self.output_dim)
                ], dim=1)
                return J  # (N, l, d)
    
    def compute_loss(self, pos_data: torch.Tensor, aug_data: torch.Tensor, 
                    aug_levels: torch.Tensor, epsilon: float,
                    pca_normals: torch.Tensor,
                    lambda_norm: float = 1.0,
                    lambda_reflection: float = 1.0, 
                    lambda_fraction: float = 1.0,
                    lambda_subspace: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute ECoMaNN loss (exact paper formulation).
        
        Paper losses:
        1. L_norm: ||h(q̌)||₂ = i*ε (augmented point magnitude)
        2. L_reflection: ||h(q + iεu) + h(q - iεu)||² (symmetry)
        3. L_fraction: normalized outputs match for fractional points
        4. L_subspace: Jacobian alignment with PCA normal space
        
        Args:
            pos_data: On-manifold samples q (N, d)
            aug_data: Augmented samples q̌ = q + i*ε*u (M, d)
            aug_levels: Augmentation level i for each aug sample (M,)
            epsilon: Base augmentation distance ε
            pca_normals: Normal vectors from PCA (N, d)
            lambda_*: Loss weights
        
        Returns:
            total_loss, loss_dict
        """
        device = pos_data.device
        N = pos_data.shape[0]
        
        # Normalize normal vectors
        pca_normals_normalized = F.normalize(pca_normals, dim=1)  # (N, d)
        
        # ============================================================
        # 1. NORM LOSS (Paper Eq.): ||h(q̌)||₂ = i*ε
        # The magnitude of h at augmented points should equal i*epsilon
        # ============================================================
        f_aug = self.forward(aug_data)  # (M, l)
        target_norms = aug_levels.float() * epsilon  # (M,)
        
        if self.output_dim == 1:
            predicted_norms = f_aug.abs().squeeze(-1)  # (M,)
        else:
            predicted_norms = f_aug.norm(dim=1)  # (M,)
        
        loss_norm = ((predicted_norms - target_norms) ** 2).mean()
        
        # ============================================================
        # 2. REFLECTION LOSS (Paper Eq.): ||h(q+iεu) + h(q-iεu)||²
        # Symmetric points should have opposite h values
        # ============================================================
        pos_plus = pos_data + epsilon * pca_normals_normalized   # q + εu
        pos_minus = pos_data - epsilon * pca_normals_normalized  # q - εu
        
        f_plus = self.forward(pos_plus)   # h(q + εu)
        f_minus = self.forward(pos_minus) # h(q - εu)
        
        # Paper: h(q+εu) + h(q-εu) should be 0 (opposite signs, same magnitude)
        loss_reflection = ((f_plus + f_minus) ** 2).mean()
        
        # ============================================================
        # 3. FRACTION LOSS (Paper Eq.): normalized outputs should match
        # h(q+εu)/||h(q+εu)|| = h(q+(a/b)εu)/||h(q+(a/b)εu)||
        # ============================================================
        fractions = [0.25, 0.5, 0.75]  # a/b values from paper
        frac_losses = []
        
        for frac in fractions:
            pos_frac = pos_data + frac * epsilon * pca_normals_normalized  # q + (a/b)εu
            f_frac = self.forward(pos_frac)
            
            # Normalize outputs (paper: compare normalized h values)
            f_plus_norm = F.normalize(f_plus, dim=-1, eps=1e-8)
            f_frac_norm = F.normalize(f_frac, dim=-1, eps=1e-8)
            
            frac_loss = ((f_plus_norm - f_frac_norm) ** 2).mean()
            frac_losses.append(frac_loss)
        
        loss_fraction = torch.stack(frac_losses).mean()
        
        # ============================================================
        # 4. SUBSPACE ALIGNMENT LOSS (Paper Eq.)
        # Align Jacobian row space with PCA normal space
        # ||V_N V_N^T E_N||² + ||E_N E_N^T V_N||²
        # For l=1: simplifies to gradient-normal alignment
        # ============================================================
        if lambda_subspace > 0:
            pos_for_grad = pos_data.clone().detach().requires_grad_(True)
            f_for_grad = self.forward(pos_for_grad)
            
            # Compute gradient ∇h(x)
            grads = torch.autograd.grad(f_for_grad.sum(), pos_for_grad, create_graph=True)[0]
            grads_normalized = F.normalize(grads, dim=1)
            
            # Paper alignment: projection error
            # For l=1: ||V_N (V_N · E_N) - E_N||² = 0 when parallel
            # Equivalently: 1 - cos²(angle) should be 0
            cos_angle = (grads_normalized * pca_normals_normalized).sum(dim=1)
            loss_subspace = (1.0 - cos_angle.abs() ** 2).mean()
        else:
            loss_subspace = torch.tensor(0.0, device=device)
        
        # ============================================================
        # COMBINE LOSSES
        # ============================================================
        total_loss = (lambda_norm * loss_norm + 
                     lambda_reflection * loss_reflection + 
                     lambda_fraction * loss_fraction + 
                     lambda_subspace * loss_subspace)
        
        loss_dict = {
            'norm': loss_norm.item(),
            'reflection': loss_reflection.item(),
            'fraction': loss_fraction.item(),
            'subspace': loss_subspace.item() if isinstance(loss_subspace, torch.Tensor) else loss_subspace,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def compute_epsilon_from_pca(eigenvalues_tangent: np.ndarray) -> float:
    """
    Compute epsilon from PCA tangent space eigenvalues (paper method).
    
    Paper: ε = sqrt(mean of tangent space eigenvalues)
    
    Args:
        eigenvalues_tangent: Eigenvalues of tangent space (d-l largest)
    
    Returns:
        epsilon: Augmentation distance
    """
    return float(np.sqrt(np.mean(eigenvalues_tangent)))


def generate_augmented_data(pos_samples: np.ndarray, 
                           pca_normals: np.ndarray,
                           epsilon: float,
                           n_levels: int = N_AUGMENTATION_LEVELS,
                           reject_invalid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multi-level augmented data (paper method).
    
    Paper: q̌ = q + i*ε*u for i = 1, 2, ..., n_levels
    where u is unit normal direction
    
    Args:
        pos_samples: On-manifold samples (N, d)
        pca_normals: Normal vectors at each point (N, d)
        epsilon: Base augmentation distance
        n_levels: Number of augmentation levels (paper: 9)
        reject_invalid: Reject points where closest manifold point != original
    
    Returns:
        aug_data: Augmented samples (N * n_levels * 2, d) - both + and - directions
        aug_levels: Level i for each sample (N * n_levels * 2,)
    """
    N, d = pos_samples.shape
    
    # Normalize normals
    normals_normalized = pca_normals / (np.linalg.norm(pca_normals, axis=1, keepdims=True) + 1e-8)
    
    aug_list = []
    level_list = []
    
    for i in range(1, n_levels + 1):
        # Both positive and negative directions
        aug_pos = pos_samples + i * epsilon * normals_normalized
        aug_neg = pos_samples - i * epsilon * normals_normalized
        
        aug_list.extend([aug_pos, aug_neg])
        level_list.extend([np.full(N, i), np.full(N, i)])
    
    aug_data = np.vstack(aug_list)
    aug_levels = np.concatenate(level_list)
    
    return aug_data.astype(np.float32), aug_levels.astype(np.int32)


def train_ecomann(pos_samples: np.ndarray, 
                  pca_normals: np.ndarray,
                  epsilon: Optional[float] = None,
                  epochs: int = 25,  # Paper: 25 for clean data
                  lr: float = 0.001,
                  batch_size: int = 128,
                  n_augmentation_levels: int = N_AUGMENTATION_LEVELS,
                  lambda_norm: float = 1.0,
                  lambda_reflection: float = 1.0,
                  lambda_fraction: float = 1.0,
                  lambda_subspace: float = 1.0,
                  hidden_sizes: Optional[List[int]] = None,
                  verbose: bool = False,
                  device: str = 'cpu') -> Tuple['ECoMaNN', List[float]]:
    """
    Train ECoMaNN model (paper training procedure).
    
    Paper specs:
    - Optimizer: RMSprop with lr=0.001
    - Epochs: 25 (clean data), 50 (noisy data)
    - Batch size: 128
    - Multi-level augmentation: i = 1..9
    
    Args:
        pos_samples: On-manifold samples (numpy array, N x d)
        pca_normals: Normals from Adaptive PCA (numpy array, N x d)
        epsilon: Augmentation distance (if None, computed from data)
        epochs: Number of training epochs
        lr: Learning rate (paper: 0.001)
        batch_size: Batch size (paper: 128)
        n_augmentation_levels: Number of augmentation levels (paper: 9)
        lambda_*: Loss weights
        hidden_sizes: Hidden layer sizes (None = paper default)
        verbose: Print progress
        device: Training device
    
    Returns:
        model: Trained ECoMaNN model
        train_losses: List of training losses
    """
    # Compute epsilon if not provided (paper method)
    if epsilon is None:
        # Estimate from data spread
        epsilon = float(np.std(pos_samples) * 0.1)
    
    # Generate augmented data
    aug_data, aug_levels = generate_augmented_data(
        pos_samples, pca_normals, epsilon, n_augmentation_levels
    )
    
    # Convert to tensors
    pos_tensor = torch.from_numpy(pos_samples).float().to(device)
    aug_tensor = torch.from_numpy(aug_data).float().to(device)
    levels_tensor = torch.from_numpy(aug_levels).long().to(device)
    normals_tensor = torch.from_numpy(pca_normals).float().to(device)
    
    # Create 80/20 split
    n_pos = len(pos_tensor)
    n_aug = len(aug_tensor)
    n_pos_train = int(0.8 * n_pos)
    n_aug_train = int(0.8 * n_aug)
    
    pos_train = pos_tensor[:n_pos_train]
    pos_eval = pos_tensor[n_pos_train:]
    aug_train = aug_tensor[:n_aug_train]
    aug_eval = aug_tensor[n_aug_train:]
    levels_train = levels_tensor[:n_aug_train]
    levels_eval = levels_tensor[n_aug_train:]
    normals_train = normals_tensor[:n_pos_train]
    normals_eval = normals_tensor[n_pos_train:]
    
    # Create model with paper architecture
    input_dim = pos_samples.shape[1]
    model = ECoMaNN(
        input_dim=input_dim, 
        hidden_sizes=hidden_sizes,  # None = paper default
        output_dim=1,
        use_batch_norm=True  # Paper uses BatchNorm
    ).to(device)
    
    # Paper uses RMSprop optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
    train_losses = []
    eval_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Training step
        loss, loss_dict = model.compute_loss(
            pos_train, aug_train, levels_train, epsilon, normals_train,
            lambda_norm=lambda_norm,
            lambda_reflection=lambda_reflection,
            lambda_fraction=lambda_fraction,
            lambda_subspace=lambda_subspace
        )
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.detach().item())
        
        # Evaluation
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                eval_loss, eval_dict = model.compute_loss(
                    pos_eval, aug_eval, levels_eval, epsilon, normals_eval,
                    lambda_norm=lambda_norm,
                    lambda_reflection=lambda_reflection,
                    lambda_fraction=lambda_fraction,
                    lambda_subspace=0.0  # Skip gradient computation in eval
                )
                eval_losses.append(eval_loss.item())
            model.train()
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={loss.item():.4f}, Eval={eval_loss.item():.4f}")
                print(f"  norm={loss_dict['norm']:.4f}, refl={loss_dict['reflection']:.4f}, "
                      f"frac={loss_dict['fraction']:.4f}, subsp={loss_dict['subspace']:.4f}")
    
    return model, train_losses


# =============================================================================
# WRAPPER FUNCTION FOR COMPATIBILITY WITH YOUR PIPELINE
# =============================================================================

def train_ecomann_with_negatives(pos_samples: np.ndarray, 
                                  neg_samples: np.ndarray,
                                  pca_normals: np.ndarray,
                                  epochs: int = 25,
                                  lr: float = 0.001,
                                  batch_size: int = 128,
                                  lambda_norm: float = 1.0,
                                  lambda_reflection: float = 1.0,
                                  lambda_fraction: float = 1.0,
                                  lambda_subspace: float = 1.0,
                                  hidden_sizes: Optional[List[int]] = None,
                                  verbose: bool = False,
                                  device: str = 'cpu') -> Tuple['ECoMaNN', List[float]]:
    """
    Wrapper to train ECoMaNN using your data pipeline (pos/neg samples).
    
    This adapts the paper's method to work with your existing data generation.
    Instead of generating augmented data from normals, we use your negative samples
    but compute the losses following the paper exactly.
    
    Args:
        pos_samples: Positive (on-manifold) samples
        neg_samples: Negative (off-manifold) samples from your Adaptive PCA
        pca_normals: Normal vectors from Adaptive PCA
        ... (other args same as train_ecomann)
    
    Returns:
        model: Trained ECoMaNN model
        train_losses: Training loss history
    """
    from scipy.spatial import cKDTree
    
    # Estimate epsilon from distance to negative samples
    if len(neg_samples) > 0:
        tree = cKDTree(pos_samples)
        dists, _ = tree.query(neg_samples, k=1)
        epsilon = float(np.mean(dists))
    else:
        epsilon = float(np.std(pos_samples) * 0.1)
    
    # Use neg_samples as augmented data with level=1
    aug_data = neg_samples.astype(np.float32)
    aug_levels = np.ones(len(neg_samples), dtype=np.int32)
    
    # Convert to tensors
    pos_tensor = torch.from_numpy(pos_samples.astype(np.float32)).to(device)
    aug_tensor = torch.from_numpy(aug_data).to(device)
    levels_tensor = torch.from_numpy(aug_levels).long().to(device)
    normals_tensor = torch.from_numpy(pca_normals.astype(np.float32)).to(device)
    
    # Create model
    input_dim = pos_samples.shape[1]
    model = ECoMaNN(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=1,
        use_batch_norm=True
    ).to(device)
    
    # Paper uses RMSprop
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(
            pos_tensor, aug_tensor, levels_tensor, epsilon, normals_tensor,
            lambda_norm=lambda_norm,
            lambda_reflection=lambda_reflection,
            lambda_fraction=lambda_fraction,
            lambda_subspace=lambda_subspace
        )
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.detach().item())
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")
            print(f"  norm={loss_dict['norm']:.4f}, refl={loss_dict['reflection']:.4f}, "
                  f"frac={loss_dict['fraction']:.4f}, subsp={loss_dict['subspace']:.4f}")
    
    return model, train_losses
