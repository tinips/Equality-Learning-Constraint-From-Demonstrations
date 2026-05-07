import torch
import torch.nn as nn

class MLPImplicit(nn.Module):
	"""
	Simple MLP for implicit function learning in 2D.
	Args:
		in_dim (int): Input dimension (default 2).
		hidden (int): Hidden layer size.
		layers (int): Number of hidden layers.
	"""
	def __init__(self, in_dim=2, hidden=16, layers=2):
		super().__init__()
		net = []
		last = in_dim
		for _ in range(layers):
			net += [nn.Linear(last, hidden), nn.Tanh()]
			last = hidden
		net += [nn.Linear(last, 1)]
		self.net = nn.Sequential(*net)
	def forward(self, x):
		return self.net(x).squeeze(-1)


class PolyImplicit(nn.Module):
	"""
	Polynomial implicit model for 2D input.
	Args:
		degree (int): Maximum degree of the polynomial.
	"""
	def __init__(self, degree=2):
		super().__init__()
		self.degree = degree
		self.M = sum(1 for d in range(degree+1) for _ in range(d+1))
		self.w = nn.Parameter(torch.randn(self.M)*0.01)
	def forward(self, x):
		Phi = poly_features(x, self.degree)
		return (Phi @ self.w).squeeze(-1)


class HybridImplicit(nn.Module):
	"""
	Hybrid implicit model: f(x) = polynomial(x) + neural_network(x)
	The neural network is initialized small so it outputs near zero initially.
	If the polynomial is sufficient, the neural network will remain near zero.
	Args:
		degree (int): Maximum degree of the polynomial component.
		hidden (int): Hidden layer size for neural network.
		layers (int): Number of hidden layers for neural network.
	"""
	def __init__(self, degree=2, hidden=16, layers=2):
		super().__init__()
		
		# Polynomial component (prior knowledge)
		self.degree = degree
		self.M = sum(1 for d in range(degree+1) for _ in range(d+1))
		self.poly_w = nn.Parameter(torch.randn(self.M)*0.01)
		
		# Neural network component (residual correction)
		net = []
		last = 2  # 2D input
		for _ in range(layers):
			net += [nn.Linear(last, hidden), nn.Tanh()]
			last = hidden
		net += [nn.Linear(last, 1)]
		self.mlp_net = nn.Sequential(*net)
		
		# Initialize neural network weights to be small but not tiny
		# This allows the network to contribute when polynomial is insufficient
		for module in self.mlp_net:
			if isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.01)
				nn.init.constant_(module.bias, 0.0)
	
	def forward(self, x):
		# Polynomial component 
		Phi = poly_features(x, self.degree)
		poly_out = (Phi @ self.poly_w).squeeze(-1)
		
		# Neural network component 
		mlp_out = self.mlp_net(x).squeeze(-1)
		
		# Sum both components
		return poly_out + mlp_out

	
def poly_features(x, degree):
	"""
	Generate polynomial features up to a given degree for 2D input.
	Args:
		x (Tensor): Input tensor of shape (N, 2)
		degree (int): Maximum degree
	Returns:
		Tensor: Feature matrix (N, num_features)
	"""
	N = x.shape[0]
	feats = [torch.ones(N, 1, dtype=x.dtype, device=x.device)]
	for d in range(1, degree+1):
		for i in range(d+1):
			j = d - i
			feats.append((x[:, :1]**i) * (x[:, 1:]**j))
	return torch.cat(feats, dim=1)