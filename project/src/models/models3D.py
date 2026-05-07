import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class AutoEncoder(nn.Module):
	def __init__(self, in_dim=3, latent_dim=2, hidden=64):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(in_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, latent_dim)
		)

		self.output = nn.Sequential(
			nn.Linear(latent_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, 1)
		)
	
	def encode(self, x):
		return self.encoder(x)

	def forward(self, x):
		z = self.encode(x)
		h = self.output(z).squeeze(-1)  
		return h


class SpiralImplicitMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=64, layers=2, out_dim=1):
        super().__init__()
        net = []
        last = in_dim

        for _ in range(layers):
            net += [nn.Linear(last, hidden), nn.ReLU()]
            last = hidden

        net += [nn.Linear(last, out_dim)]

        self.net = nn.Sequential(*net)
				
    def forward(self, xyz):  
        return self.net(xyz).squeeze(-1)  


class Polynomial3D(nn.Module):
	def __init__(self, in_dim=3, degree=2):
		super().__init__()
		self.degree = degree
		self.in_dim = in_dim
		
		n_features = self._calculate_features_count()
		self.linear = nn.Linear(n_features, 1)
	
	def _calculate_features_count(self):
		count = 0
		
		count += self.in_dim  
		
		if self.degree >= 2:
			count += self.in_dim

			count += 3  
		
		if self.degree >= 3:
			count += self.in_dim 
			
			count += 1  

		if self.degree >= 4:
			count += self.in_dim  
		
		if self.degree >= 5:
			count += self.in_dim  
		
		if self.degree >= 6:
			count += self.in_dim  
			
		return count
	
	def polynomial_features(self, x):
		features = []
		
		features.append(x)  
		
		if self.degree >= 2:
			features.append(x ** 2)

			xy = (x[:, 0] * x[:, 1]).unsqueeze(1)
			xz = (x[:, 0] * x[:, 2]).unsqueeze(1)  
			yz = (x[:, 1] * x[:, 2]).unsqueeze(1)
			features.extend([xy, xz, yz])

		if self.degree >= 3:
			features.append(x ** 3)

			xyz = (x[:, 0] * x[:, 1] * x[:, 2]).unsqueeze(1)
			features.append(xyz)
		
		if self.degree >= 4:
			features.append(x ** 4)

		if self.degree >= 5:
			features.append(x ** 5)

		if self.degree >= 6:
			features.append(x ** 6)

		return torch.cat(features, dim=1)
	
	def forward(self, x):
		poly_x = self.polynomial_features(x)
		return self.linear(poly_x).squeeze(-1)


class ParametricQuad(nn.Module):
	"""Minimal parametric quadratic: h = a*x^2 + b*y^2 + c*z^2 - 1.

	This implementation is intentionally minimal and deterministic:
	- a, b, c are torch.nn.Parameter
	- bias is fixed at -1
	"""
	def __init__(self, init_a=1.0, init_b=1.0, init_c=1.0):
		super().__init__()
		self.a = nn.Parameter(torch.tensor(float(init_a), dtype=torch.float32))
		self.b = nn.Parameter(torch.tensor(float(init_b), dtype=torch.float32))
		self.c = nn.Parameter(torch.tensor(float(init_c), dtype=torch.float32))
		self.register_buffer('_bias_const', torch.tensor(-1.0, dtype=torch.float32))

	def forward(self, x):
		if not torch.is_tensor(x):
			x = torch.tensor(x, dtype=torch.float32)
		xsq = x[:, 0] ** 2
		ysq = x[:, 1] ** 2
		zsq = x[:, 2] ** 2
		h = self.a * xsq + self.b * ysq + self.c * zsq + self._bias_const
		return h.squeeze(-1)


class ParametricLinear(nn.Module):
	"""Minimal parametric linear: h = a*x + b*y + c*z + bias.

	- a, b, c are torch.nn.Parameter
	- bias is fixed at 0 (for hyperplanes through origin)
	"""
	def __init__(self, init_a=0.0, init_b=1.0, init_c=-1.0):
		super().__init__()
		self.a = nn.Parameter(torch.tensor(float(init_a), dtype=torch.float32))
		self.b = nn.Parameter(torch.tensor(float(init_b), dtype=torch.float32))
		self.c = nn.Parameter(torch.tensor(float(init_c), dtype=torch.float32))
		self.register_buffer('_bias_const', torch.tensor(0.0, dtype=torch.float32))

	def forward(self, x):
		if not torch.is_tensor(x):
			x = torch.tensor(x, dtype=torch.float32)
		h = self.a * x[:, 0] + self.b * x[:, 1] + self.c * x[:, 2] + self._bias_const
		return h.squeeze(-1)




class AnalyticalSpiral(nn.Module):
	def __init__(self, R=0.5, init_alpha=0.3, init_theta=0.0, tube_radius=0.05):
		super().__init__()
		self.R = nn.Parameter(torch.tensor(float(R), dtype=torch.float32), requires_grad=True)
		self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))
		self.theta_offset = nn.Parameter(torch.tensor(float(init_theta), dtype=torch.float32))
		self.tube_radius = float(tube_radius)

	def forward(self, x):
		if not torch.is_tensor(x):
			x = torch.tensor(x, dtype=torch.float32)
		if x.ndim == 1:
			x = x.unsqueeze(0)

		px = x[:, 0]
		py = x[:, 1]
		pz = x[:, 2]

		alpha_safe = self.alpha + 1e-8
		t = pz / alpha_safe

		phi = t + self.theta_offset
		Cx = self.R * torch.cos(phi)
		Cy = self.R * torch.sin(phi)
		Cz = self.alpha * t

		dx = px - Cx
		dy = py - Cy
		dz = pz - Cz
		dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
		return (dist - self.tube_radius).squeeze(-1)

	def sample_curve(self, t):
		if not torch.is_tensor(t):
			t = torch.tensor(t, dtype=torch.float32)
		phi = t + self.theta_offset
		x = self.R * torch.cos(phi)
		y = self.R * torch.sin(phi)
		z = self.alpha * t
		return torch.stack([x, y, z], dim=-1)



class HybridModel3D(nn.Module):
	def __init__(self, model1, model2, switch_z=2.0, blend_width=0.5):
		"""
		Hybrid model with smooth blending between two models.
		
		Args:
			model1: Model for lower z values
			model2: Model for higher z values
			switch_z: Center of transition region
			blend_width: Width of smooth transition (higher = smoother, lower = sharper)
		"""
		super().__init__()
		self.model1 = model1
		self.model2 = model2
		self.switch_z = switch_z
		self.blend_width = blend_width

	def forward(self, x):
		z = x[:, 2]
		
		# Smooth blending using sigmoid (differentiable)
		# When z << switch_z: weight2 ≈ 0 (use model1)
		# When z >> switch_z: weight2 ≈ 1 (use model2)
		# At z = switch_z: weight2 = 0.5 (average both)
		weight2 = torch.sigmoid((z - self.switch_z) / self.blend_width)
		weight1 = 1.0 - weight2

		h1 = self.model1(x)
		h2 = self.model2(x)

		# Smooth interpolation (gradients flow to both models)
		h = weight1 * h1 + weight2 * h2
		return h