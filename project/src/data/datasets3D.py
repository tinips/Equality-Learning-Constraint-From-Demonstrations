import numpy as np
import math
import torch
import random
from src.utils.seed import set_seed

def make_constraint_3d(name: str, grid_lim=2.5, **kwargs):
	"""
	Create 3D implicit constraint functions for various shapes.
	Similar interface to make_constraint but for 3D shapes.
	
	Args:
		name: Shape name ("sphere", "spiral", "cylinder", "plane", etc.)
		grid_lim: Spatial limits for filtering samples
		**kwargs: Shape-specific parameters
		
	Returns:
		h_np: Implicit function h(P) - signed distance
		grad_np: Gradient function (normalized)
		gt_samples: Ground truth samples on surface
		gt_curve: Alias for gt_samples (consistency with 2D)
	"""
	name = name.lower()
	
	if name == "sphere":
		R = kwargs.get('radius', 1.0)
		center = kwargs.get('center', (0.0, 0.0, 0.0))
		c = np.array(center, dtype=np.float64).reshape(1, 3)
		
		def h_np(P):
			P = np.asarray(P, dtype=np.float64)
			d = P - c
			r = np.linalg.norm(d, axis=1)
			return r - R
			
		def grad_np(P):
			P = np.asarray(P, dtype=np.float64)
			d = P - c
			r = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
			return d / r
			
		def gt_samples(n=25000):
			u = np.random.normal(size=(n, 3))
			u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
			pts = c + R * u
			mask = (np.abs(pts) <= grid_lim).all(axis=1)
			return pts[mask]
			
		def gt_curve(M=5000):
			return gt_samples(M)
			
	elif name == "ellipsoid":
		a = kwargs.get('a', 1.5)  # Semi-axis X
		b = kwargs.get('b', 1.0)  # Semi-axis Y
		c = kwargs.get('c', 0.8)  # Semi-axis Z
		center = kwargs.get('center', (0.0, 0.0, 0.0))
		cx, cy, cz = center
		
		def h_np(P):
			P = np.asarray(P, dtype=np.float64)
			if P.ndim == 1:
				P = P.reshape(1, 3)
			x, y, z = P[:, 0] - cx, P[:, 1] - cy, P[:, 2] - cz
			# Implicit function: (x/a)^2 + (y/b)^2 + (z/c)^2 - 1 = 0
			val = (x/a)**2 + (y/b)**2 + (z/c)**2 - 1.0
			return val
			
		def grad_np(P):
			P = np.asarray(P, dtype=np.float64)
			if P.ndim == 1:
				P = P.reshape(1, 3)
			x, y, z = P[:, 0] - cx, P[:, 1] - cy, P[:, 2] - cz
			# Gradient: ∇f = (2x/a^2, 2y/b^2, 2z/c^2)
			grads = np.zeros_like(P)
			grads[:, 0] = 2.0 * x / (a * a)
			grads[:, 1] = 2.0 * y / (b * b)
			grads[:, 2] = 2.0 * z / (c * c)
			# Normalize
			norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-9
			return grads / norms
			
		def gt_samples(n=25000):
			# Sample uniformly on ellipsoid surface using spherical coordinates
			u = np.random.uniform(0, 2*np.pi, n)
			v = np.random.uniform(0, np.pi, n)
			x = cx + a * np.sin(v) * np.cos(u)
			y = cy + b * np.sin(v) * np.sin(u)
			z = cz + c * np.cos(v)
			pts = np.stack([x, y, z], axis=1)
			mask = (np.abs(pts) <= grid_lim).all(axis=1)
			return pts[mask]
			

		def sample_trajectory(total_length, num_steps, v_pref=None, grid_lim=grid_lim, seed=None):
			"""
			Generate a continuous trajectory along an approximately constant-latitude
		tring on the ellipsoid surface.
			Returns `num_steps` points with approximate arc length `total_length`.
			"""
			# Use a random initial point and a random tangent direction (seeded) so
			# trajectories explore all directions (including diagonal/vertical).
			# Local RNG for reproducibility
			if seed is not None:
				rng = np.random.default_rng(int(seed))
			else:
				rng = np.random

			# pick an initial parameter (u0, v0) uniformly on parameter domain (avoid poles)
			v0 = float(rng.uniform(0.05 * np.pi, 0.95 * np.pi))
			u0 = float(rng.uniform(0.0, 2.0 * np.pi))

			# initial point on ellipsoid
			x0p = cx + a * np.sin(v0) * np.cos(u0)
			y0p = cy + b * np.sin(v0) * np.sin(u0)
			z0p = float(cz + c * np.cos(v0))
			P_initial = np.array([x0p, y0p, z0p], dtype=float)

			# desired step length along curve
			step_size = float(total_length) / float(num_steps)

			# compute normal at P_initial (unnormalized gradient then normalize)
			grad_init = np.array([
				2.0 * (P_initial[0] - cx) / (a * a),
				2.0 * (P_initial[1] - cy) / (b * b),
				2.0 * (P_initial[2] - cz) / (c * c)
			], dtype=float)
			norm_grad = np.linalg.norm(grad_init) + 1e-12
			N_initial = grad_init / norm_grad

			# sample random tangent orthogonal to normal
			def _random_tangent_dir(nvec, rng_local):
				for _ in range(20):
					r = rng_local.normal(size=3)
					r = r - np.dot(r, nvec) * nvec
					n = np.linalg.norm(r)
					if n > 1e-8:
						return r / n
				# fallback deterministic
				z_axis = np.array([0.0, 0.0, 1.0])
				t_tmp = np.cross(nvec, z_axis)
				if np.linalg.norm(t_tmp) < 1e-6:
					x_axis = np.array([1.0, 0.0, 0.0])
					t_tmp = np.cross(nvec, x_axis)
				return t_tmp / (np.linalg.norm(t_tmp) + 1e-9)

			T_initial = _random_tangent_dir(N_initial, rng)

			# perform forward integration along tangent with projection to surface
			traj_pts = [P_initial.copy()]
			P_current = P_initial.copy()
			T_current = T_initial.copy()
			max_retries = 5
			newton_iters = 50

			for i in range(num_steps - 1):
				success = False
				trial_step = step_size
				for attempt in range(max_retries):
					P_guess = P_current + T_current * trial_step
					# Newton projection onto ellipsoid: use analytic gradient (unnormalized)
					P_corr = P_guess.copy()
					for _ in range(newton_iters):
						# compute implicit function value f(P) = (x-cx)^2/a^2 + ... - 1
						x, y, z = P_corr[0] - cx, P_corr[1] - cy, P_corr[2] - cz
						fval = (x / a)**2 + (y / b)**2 + (z / c)**2 - 1.0
						if abs(fval) < 1e-9:
							break
						# gradient of f
						gradf = np.array([2.0 * x / (a * a), 2.0 * y / (b * b), 2.0 * z / (c * c)])
						# Newton step along gradf
						denom = np.dot(gradf, gradf)
						if denom <= 1e-12:
							break
						P_corr = P_corr - (fval / denom) * gradf
					# validate
					x, y, z = P_corr
					if (np.abs(P_corr) <= grid_lim).all():
						# accept if close to surface
						# recompute fval
						xp, yp, zp = P_corr[0] - cx, P_corr[1] - cy, P_corr[2] - cz
						fval2 = (xp / a)**2 + (yp / b)**2 + (zp / c)**2 - 1.0
						if abs(fval2) <= 1e-3:
							success = True
							break
					# shrink step and retry
					trial_step *= 0.5
					if trial_step < 1e-6:
						break
				if not success:
					# cannot advance further
					break
				# update tangent: project onto new tangent plane
				# compute new normal (unnormalized) and normalize
				xg, yg, zg = P_corr[0] - cx, P_corr[1] - cy, P_corr[2] - cz
				grad_new = np.array([2.0 * xg / (a * a), 2.0 * yg / (b * b), 2.0 * zg / (c * c)])
				N_new = grad_new / (np.linalg.norm(grad_new) + 1e-12)
				T_next = T_current - np.dot(T_current, N_new) * N_new
				norm_T = np.linalg.norm(T_next)
				if norm_T < 1e-6:
					# fallback tangent
					x_axis = np.array([1.0, 0.0, 0.0])
					t_tmp = np.cross(N_new, x_axis)
					if np.linalg.norm(t_tmp) < 1e-6:
						y_axis = np.array([0.0, 1.0, 0.0])
						t_tmp = np.cross(N_new, y_axis)
					T_next = t_tmp / (np.linalg.norm(t_tmp) + 1e-9)
				else:
					T_next = T_next / (norm_T + 1e-9)
				# maintain continuity
				if np.dot(T_next, T_current) < 0:
					T_next = -T_next
				traj_pts.append(P_corr)
				P_current = P_corr
				T_current = T_next

			# if we have fewer than requested points, pad by repeating last
			if len(traj_pts) < num_steps:
				while len(traj_pts) < num_steps:
					traj_pts.append(traj_pts[-1].copy())
			return np.array(traj_pts[:num_steps])

		def gt_curve(M=5000):
			return gt_samples(M)

		# attach the specialized sampler so callers can request ellipsoid trajectories
		gt_samples.sample_trajectory = sample_trajectory
	
	elif name == "spiral":
		R = kwargs.get('R0', 1.0)           
		alpha = kwargs.get('pitch', 0.3) 
		t_min = kwargs.get('t_min', -2*np.pi)  
		t_max = kwargs.get('t_max', 2*np.pi)   
		t_samples = kwargs.get('t_samples', 300)  

		t_grid = np.linspace(t_min, t_max, t_samples)

		def centerline(t):
			return np.stack([R * np.cos(t), R * np.sin(t), alpha * t], axis=1)

		def h_np(P):
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			# Distance to all centerline points, take minimum
			C = centerline(t_grid)
			diff = P[:, None, :] - C[None, :, :]  
			dists = np.linalg.norm(diff, axis=2)  
			return np.min(dists, axis=1) 

		def grad_np(P, eps=1e-4):
			"""Numerical gradient (direction away from spiral)"""
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			grads = np.zeros_like(P)
			for i in range(3):
				d = np.zeros(3); d[i] = eps
				gp = h_np(P + d)
				gm = h_np(P - d)
				grads[:, i] = (gp - gm) / (2 * eps)
			# normalize
			norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
			return grads / norms

		def gt_samples(n=1000):
			"""Sample points ON the spiral centerline"""
			t = np.random.uniform(t_min, t_max, size=(n,))
			pts = centerline(t)
			mask = (np.abs(pts) <= grid_lim).all(axis=1)
			return pts[mask]

		def gt_curve(M=1000):
			"""Dense sampling along spiral for visualization"""
			t = np.linspace(t_min, t_max, M)
			return centerline(t)

	elif name == "spiral_tube":
		"""
		Spiral tube: a tube (3D surface) wrapping around in a helix.
		Like a spring/coil with thickness.
		"""
		R = kwargs.get('R0', 1.0)           # Radius of spiral centerline
		alpha = kwargs.get('pitch', 0.3)    # Vertical pitch
		r_tube = kwargs.get('r_tube', 0.4) # Tube thickness radius
		t_min = kwargs.get('t_min', -2*np.pi)
		t_max = kwargs.get('t_max', 2*np.pi)
		t_samples = kwargs.get('t_samples', 300)

		t_grid = np.linspace(t_min, t_max, t_samples)

		def centerline(t):
			"""Centerline of the spiral tube"""
			return np.stack([R * np.cos(t), R * np.sin(t), alpha * t], axis=1)

		def h_np(P):
			"""Signed distance to tube surface"""
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			# Find distance to centerline
			C = centerline(t_grid)
			diff = P[:, None, :] - C[None, :, :]
			dists_to_centerline = np.linalg.norm(diff, axis=2)
			min_dist_to_centerline = np.min(dists_to_centerline, axis=1)
			# Signed distance to tube surface
			return min_dist_to_centerline - r_tube

		def grad_np(P, eps=1e-4):
			"""Numerical gradient pointing away from tube surface"""
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			grads = np.zeros_like(P)
			for i in range(3):
				d = np.zeros(3); d[i] = eps
				gp = h_np(P + d)
				gm = h_np(P - d)
				grads[:, i] = (gp - gm) / (2 * eps)
			norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
			return grads / norms

		def gt_samples(n=2000):
			"""Sample points ON the tube surface"""
			# Sample parameter t along spiral
			t = np.random.uniform(t_min, t_max, size=(n,))
			# Sample angle theta around tube
			theta = np.random.uniform(0, 2*np.pi, size=(n,))
			
			# Get centerline points
			center_pts = centerline(t)
			
			# Compute local frame at each point
			# Tangent direction
			tangent_x = -R * np.sin(t)
			tangent_y = R * np.cos(t)
			tangent_z = np.full_like(t, alpha)
			tangent_norm = np.sqrt(tangent_x**2 + tangent_y**2 + tangent_z**2)
			tangent_x /= tangent_norm
			tangent_y /= tangent_norm
			tangent_z /= tangent_norm
			
			# Radial direction (from Z-axis to point on xy circle)
			radial_x = np.cos(t)
			radial_y = np.sin(t)
			radial_z = np.zeros_like(t)
			
			# Binormal direction (cross product of tangent and radial)
			binormal_x = tangent_y * radial_z - tangent_z * radial_y
			binormal_y = tangent_z * radial_x - tangent_x * radial_z
			binormal_z = tangent_x * radial_y - tangent_y * radial_x
			binormal_norm = np.sqrt(binormal_x**2 + binormal_y**2 + binormal_z**2) + 1e-12
			binormal_x /= binormal_norm
			binormal_y /= binormal_norm
			binormal_z /= binormal_norm
			
			# Offset from centerline in circular pattern around tube
			offset_x = r_tube * (np.cos(theta) * radial_x + np.sin(theta) * binormal_x)
			offset_y = r_tube * (np.cos(theta) * radial_y + np.sin(theta) * binormal_y)
			offset_z = r_tube * (np.cos(theta) * radial_z + np.sin(theta) * binormal_z)
			
			pts = center_pts + np.column_stack([offset_x, offset_y, offset_z])
			mask = (np.abs(pts) <= grid_lim).all(axis=1)
			return pts[mask]

		def gt_curve(M=1000):
			"""Return tube surface samples for visualization (not just centerline)"""
			return gt_samples(M)

	elif name == "cylinder":
		R = kwargs.get('radius', 1.0)
		H = kwargs.get('height', 4.0)
		def h_np(P):
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			x = P[:, 0]
			y = P[:, 1]
			r_xy = np.sqrt(x * x + y * y)
			return r_xy - R

		def grad_np(P, eps=1e-12):
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			x = P[:, 0]
			y = P[:, 1]
			r_xy = np.sqrt(x * x + y * y) + 1e-12
			grads = np.zeros_like(P)
			grads[:, 0] = x / r_xy
			grads[:, 1] = y / r_xy
			grads[:, 2] = 0.0
			norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
			return grads / norms

		def gt_samples(n=1000):
			theta = np.random.uniform(0.0, 2.0 * np.pi, size=(n,))
			z = np.random.uniform(-H/2.0, H/2.0, size=(n,))
			x = R * np.cos(theta)
			y = R * np.sin(theta)
			pts = np.stack([x, y, z], axis=1)
			mask = (np.abs(pts) <= grid_lim).all(axis=1)
			return pts[mask]

		def gt_curve(M=5000):
			return gt_samples(M)

	elif name == "hyperplane":
		n_raw = np.array((0.0, 1.0, -1.0), dtype=np.float64)
		d_raw = 0.0

		norm = np.linalg.norm(n_raw) + 1e-9
		normal = n_raw / norm
		d = d_raw / norm

		def h_np(P):
			P = np.asarray(P, dtype=np.float64)
			if P.ndim == 1:
				P = P.reshape(1, 3)
			return (P @ normal) - d

		def grad_np(P):
			P = np.asarray(P, dtype=np.float64)
			if P.ndim == 1:
				P = P.reshape(1, 3)

			return np.tile(normal, (P.shape[0], 1))

		def gt_samples(n=25000):
			x = np.random.uniform(-grid_lim, grid_lim, size=(n, 1))
			y = np.random.uniform(-grid_lim, grid_lim, size=(n, 1))
			absn = np.abs(normal)
			if absn[2] > 1e-6:
				z = (d - normal[0]*x - normal[1]*y) / normal[2]
				points = np.hstack([x, y, z])
			elif absn[1] > 1e-6:
				z = np.random.uniform(-grid_lim, grid_lim, size=(n, 1))
				y = (d - normal[0]*x - normal[2]*z) / normal[1]
				points = np.hstack([x, y, z])
			else:
				y = np.random.uniform(-grid_lim, grid_lim, size=(n, 1))
				z = np.random.uniform(-grid_lim, grid_lim, size=(n, 1))
				x = (d - normal[1]*y - normal[2]*z) / normal[0]
				points = np.hstack([x, y, z])
			mask = (np.abs(points) <= grid_lim).all(axis=1)
			return points[mask]

		def gt_curve(M=5000):
			return gt_samples(M)

	elif name == "hyperboloid":
		# Hyperboloid of one sheet: hourglass/cooling tower shape
		# Implicit equation: x²/a² + y²/b² - z²/c² = 1
		# More interesting than cylinder, moderate complexity
		a = kwargs.get('a', 1.0)  # Semi-axis X
		b = kwargs.get('b', 1.0)  # Semi-axis Y
		c = kwargs.get('c', 1.2)  # Semi-axis Z (controls "waist" tightness)
		center = kwargs.get('center', (0.0, 0.0, 0.0))
		cx, cy, cz = center
		z_lim = kwargs.get('z_lim', 2.0)  # Vertical extent
		
		def h_np(P):
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			x = P[:, 0] - cx
			y = P[:, 1] - cy
			z = P[:, 2] - cz
			# Implicit function: x²/a² + y²/b² - z²/c² - 1 = 0
			val = (x/a)**2 + (y/b)**2 - (z/c)**2 - 1.0
			return val
		
		def grad_np(P, eps=1e-12):
			P = np.atleast_2d(np.asarray(P, dtype=np.float64))
			x = P[:, 0] - cx
			y = P[:, 1] - cy
			z = P[:, 2] - cz
			# Gradient: ∇f = (2x/a², 2y/b², -2z/c²)
			grads = np.zeros_like(P)
			grads[:, 0] = 2.0 * x / (a * a)
			grads[:, 1] = 2.0 * y / (b * b)
			grads[:, 2] = -2.0 * z / (c * c)
			# Normalize
			norms = np.linalg.norm(grads, axis=1, keepdims=True) + eps
			return grads / norms
		
		def gt_samples(n=2000):
			"""Sample points ON the hyperboloid surface"""
			# Parametric form:
			# x = a * sqrt(1 + v²) * cos(u)
			# y = b * sqrt(1 + v²) * sin(u)
			# z = c * v
			# where u ∈ [0, 2π], v ∈ [-v_max, v_max]
			
			u = np.random.uniform(0, 2*np.pi, n)
			v = np.random.uniform(-z_lim/c, z_lim/c, n)
			
			sqrt_term = np.sqrt(1.0 + v**2)
			x = cx + a * sqrt_term * np.cos(u)
			y = cy + b * sqrt_term * np.sin(u)
			z = cz + c * v
			
			pts = np.stack([x, y, z], axis=1)
			mask = (np.abs(pts) <= grid_lim).all(axis=1)
			return pts[mask]
		
		def gt_curve(M=5000):
			"""Return hyperboloid surface samples for visualization"""
			return gt_samples(M)
			
	else:
		raise ValueError(f"Unknown 3D constraint: {name}")
		
	return h_np, grad_np, gt_samples, gt_curve

def sample_positive_3d(gt_samples_fn, n):
	"""Sample positive points on the 3D surface"""
	return gt_samples_fn(n)


def compute_global_curvature(pos_pts):
	"""
	Compute global geometry characteristics using PCA (MATLAB-style).
	
	Returns metrics about overall data distribution to guide local analysis.
	
	Args:
		pos_pts: Points on surface (N, 3)
		
	Returns:
		dict with:
			- planarity_ratio: eigenvalue_min / eigenvalue_max (small = planar)
			- sphericity: how uniform eigenvalues are (1.0 = perfect sphere)
			- eigenvalues: sorted eigenvalues [largest, medium, smallest]
			- suggested_k: recommended k_neighbors for local PCA
	"""
	# Global PCA (like MATLAB: center data, compute covariance, eigendecomposition)
	centered = pos_pts - pos_pts.mean(axis=0)
	cov = centered.T @ centered / len(pos_pts)
	eigenvalues = np.linalg.eigvalsh(cov)  # Sorted ascending
	eigenvalues = eigenvalues[::-1]  # Descending [max, med, min]
	
	# Planarity ratio: small = planar surface, large = spherical/volumetric
	planarity_ratio = eigenvalues[2] / (eigenvalues[0] + 1e-9)
	
	# Sphericity: how similar are the eigenvalues (1.0 = perfect sphere)
	sphericity = eigenvalues[2] / (eigenvalues[0] + 1e-9)
	
	# Adaptive k suggestion based on global geometry
	if planarity_ratio < 0.01:  # Very planar (e.g., flat surface, cylinder)
		suggested_k = 25  # Use more neighbors for stability
	elif planarity_ratio < 0.05:  # Moderately planar (e.g., ellipsoid with high eccentricity)
		suggested_k = 15
	elif planarity_ratio < 0.3:  # Curved surface (e.g., regular ellipsoid)
		suggested_k = 12
	else:  # High curvature or spherical (e.g., sphere, complex shapes)
		suggested_k = 10
	
	return {
		'planarity_ratio': planarity_ratio,
		'sphericity': sphericity,
		'eigenvalues': eigenvalues,
		'suggested_k': suggested_k
	}


def estimate_local_normals(pos_pts, k_neighbors=10):
	"""
	Estimate surface normals using local PCA (NO ground truth needed).
	
	For each point, find its k nearest neighbors and compute PCA.
	The normal is the eigenvector with smallest eigenvalue (minimal variance direction).
	
	Args:
		pos_pts: Points on surface (N, 3)
		k_neighbors: Number of neighbors for local PCA
		
	Returns:
		normals: Estimated normals (N, 3), normalized
	"""
	from sklearn.neighbors import NearestNeighbors
	
	n = pos_pts.shape[0]
	normals = np.zeros((n, 3))
	
	# Find k nearest neighbors for each point
	k = min(k_neighbors, n - 1)
	nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(pos_pts)
	distances, indices = nbrs.kneighbors(pos_pts)
	
	for i in range(n):
		# Get neighbors (excluding the point itself)
		neighbor_idx = indices[i, 1:]  # Skip first (itself)
		neighbors = pos_pts[neighbor_idx]
		
		# Center the neighbors
		centroid = neighbors.mean(axis=0)
		centered = neighbors - centroid
		
		# PCA: eigenvector with smallest eigenvalue is the normal
		cov = centered.T @ centered
		eigenvalues, eigenvectors = np.linalg.eigh(cov)
		
		# Smallest eigenvalue → normal direction
		normal = eigenvectors[:, 0]
		
		# Normalize
		normal = normal / (np.linalg.norm(normal) + 1e-9)
		
		normals[i] = normal
	
	return normals


def estimate_local_normals_adaptive(pos_pts, k_neighbors=None, verbose=False):
	"""
	IMPROVED normal estimation using GLOBAL PCA to determine optimal k_neighbors.
	
	Combines global geometry analysis (MATLAB-style PCA) with local PCA:
	1. Analyze global structure to understand surface type
	2. Choose k_neighbors adaptively based on curvature
	3. Estimate local normals with optimal k
	
	This merges concepts from MATLAB global PCA with local surface analysis.
	
	Args:
		pos_pts: Points on surface (N, 3)
		k_neighbors: Number of neighbors (None = auto-detect from global PCA)
		verbose: Print diagnostic info about global geometry
		
	Returns:
		normals: Estimated normals (N, 3), normalized
	"""
	# Step 1: Global PCA analysis (like MATLAB)
	global_info = compute_global_curvature(pos_pts)
	
	# Step 2: Use suggested k if not provided
	if k_neighbors is None:
		k_neighbors = global_info['suggested_k']
	
	if verbose:
		print(f"Global PCA Analysis:")
		print(f"  Eigenvalues: {global_info['eigenvalues']}")
		print(f"  Planarity ratio: {global_info['planarity_ratio']:.6f}")
		print(f"  Suggested k_neighbors: {global_info['suggested_k']}")
		print(f"  Using k_neighbors: {k_neighbors}")
	
	# Step 3: Local PCA with adaptive k
	normals = estimate_local_normals(pos_pts, k_neighbors=k_neighbors)
	
	return normals


def sample_negative_reference_pca(pos_pts, std=0.15, k_neighbors=10, whiten=False, use_constant_displacement=False):
	"""
	Reference Paper PCA method - exact implementation from MATLAB code.
	
	Following the reference paper's approach:
	1. Center data: xzm = x - mean(x)
	2. Compute covariance: Sigma = cov(x)
	3. Eigen decomposition: [Q,D] = eig(Sigma)
	4. Sort by descending eigenvalues
	5. Project: y = Q' * xzm
	6. Optional whitening: z = D^(-1/2) * Q' * xzm
	7. Sample negatives along smallest eigenvector (normal direction)
	
	Args:
		pos_pts: Positive samples on surface (N, 3)
		std: Standard deviation for displacement magnitude
		k_neighbors: Number of neighbors for local PCA (paper uses local neighborhoods)
		whiten: If True, apply whitening (D^(-1/2) normalization)
		use_constant_displacement: If True, use constant magnitude displacement (std) instead of random N(0, std²)
		
	Returns:
		Negative samples (N, 3)
	"""
	from sklearn.neighbors import NearestNeighbors
	
	n_pts = pos_pts.shape[0]
	k = min(k_neighbors, n_pts - 1)
	
	# Find k nearest neighbors for each point
	nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(pos_pts)
	_, indices = nbrs.kneighbors(pos_pts)
	
	normals = np.zeros_like(pos_pts)
	
	for i in range(n_pts):
		neighbors_idx = indices[i, 1:]  # Exclude self
		neighbors = pos_pts[neighbors_idx]
		
		# Step 1: Center the data (muhat = mean(x,2))
		mu_hat = np.mean(neighbors, axis=0, keepdims=True)
		xzm = neighbors - mu_hat
		
		# Step 2: Compute covariance (Sigmahat = cov(x'))
		Sigma_hat = np.cov(xzm.T)
		
		# Step 3: Eigen decomposition ([Q,D] = eig(Sigmahat))
		D, Q = np.linalg.eigh(Sigma_hat)
		
		# Step 4: Sort by descending eigenvalues ([d,ind] = sort(diag(D),'descend'))
		ind = np.argsort(D)[::-1]
		D = D[ind]
		Q = Q[:, ind]
		
		# Step 5: The normal is the eigenvector with smallest eigenvalue (last column after sorting)
		# This corresponds to the direction with minimal variance
		normal = Q[:, -1]
		
		# Ensure consistent orientation (optional - can be removed if signs don't matter)
		if np.dot(normal, normal) < 0:
			normal = -normal
			
		normals[i] = normal
	# Normalize normals
	normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
	
	if use_constant_displacement:
		# Constant displacement magnitude with random signs (fair comparison)
		signs = np.random.choice([-1, 1], size=(n_pts, 1))
		neg_pts = pos_pts + signs * std * normals
	else:
		# Random displacement (original behavior)
		eps = np.abs(np.random.normal(0.0, std, size=(n_pts, 1)))
		signs = np.random.choice([-1, 1], size=(n_pts, 1))
		eps = eps * signs
		neg_pts = pos_pts + eps * normals
	
	return neg_pts


def sample_negative_from_estimated_normals(pos_pts, std=0.15, k_neighbors=None, use_adaptive=True, use_constant_displacement=False):
	"""
	REALISTIC negative sampling using ESTIMATED normals (NO ground truth).
	
	Strategy:
	1. (Optional) Use global PCA to determine optimal k_neighbors adaptively
	2. Estimate local normals using PCA on k-nearest neighbors
	3. Displace each positive point along its estimated normal
	4. Random sign (inside/outside surface)
	
	This approximates the ground-truth method but only uses available positive samples.
	
	Args:
		pos_pts: Positive samples on surface (N, 3)
		std: Standard deviation for displacement magnitude
		k_neighbors: Number of neighbors (None = auto from global PCA if use_adaptive=True, else 10)
		use_adaptive: If True, uses global PCA to determine k adaptively
		use_constant_displacement: If True, use constant magnitude displacement (std) instead of random N(0, std²)
		
	Returns:
		Negative samples (M, 3) where M ≈ N
	"""
	# Estimate normals - adaptively or with fixed k
	if use_adaptive and k_neighbors is None:
		normals = estimate_local_normals_adaptive(pos_pts, k_neighbors=None, verbose=False)
	else:
		if k_neighbors is None:
			k_neighbors = 10
		normals = estimate_local_normals(pos_pts, k_neighbors=k_neighbors)
	
	if use_constant_displacement:
		# Constant displacement magnitude with random signs (fair comparison)
		signs = np.random.choice([-1, 1], size=(pos_pts.shape[0], 1))
		neg_pts = pos_pts + signs * std * normals
	else:
		# Random displacement (original behavior)
		eps = np.abs(np.random.normal(0.0, std, size=(pos_pts.shape[0], 1)))
		signs = np.random.choice([-1, 1], size=(pos_pts.shape[0], 1))
		eps = eps * signs
		neg_pts = pos_pts + eps * normals
	
	return neg_pts



def sample_negative_random_direction(pos_pts, std=0.15, use_constant_displacement=False):
	"""
	Random Direction negative sampling (baseline).
	Sample negatives in completely random directions from positive points.
	
	Args:
		pos_pts: Positive samples on surface (N, 3)
		std: Standard deviation for displacement magnitude
		use_constant_displacement: If True, use constant magnitude displacement (std) instead of random N(0, std²)
		
	Returns:
		Negative samples (N, 3)
	"""
	n = pos_pts.shape[0]
	
	# Generate random unit directions (isotropic)
	random_dirs = np.random.normal(size=(n, 3))
	random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-9)
	
	if use_constant_displacement:
		# Constant displacement magnitude with random signs (fair comparison)
		signs = np.random.choice([-1, 1], size=(n, 1))
		neg_pts = pos_pts + signs * std * random_dirs
	else:
		# Random displacement (original behavior)
		eps = np.abs(np.random.normal(0.0, std, size=(n, 1)))
		signs = np.random.choice([-1, 1], size=(n, 1))
		eps = eps * signs
		neg_pts = pos_pts + eps * random_dirs
	
	return neg_pts
def sample_negative_from_normals_3d(pos_pts, grad_gt_np, h_gt_np, std=0.15, use_constant_displacement=False):
	"""
	Generate negative samples using normal directions.
	Samples BOTH inside and outside the surface (50% each direction).
	
	Args:
		use_constant_displacement: If True, use constant magnitude displacement (std) instead of random N(0, std²)
	"""
	P = pos_pts
	N = grad_gt_np(P)
	# Normalize normals
	N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-8)
	
	if use_constant_displacement:
		# Constant displacement magnitude with random signs (fair comparison)
		signs = np.random.choice([-1, 1], size=(P.shape[0], 1))
		P_displaced = P + signs * std * N
		# NO filter - return all negatives for fair evaluation
		return P_displaced
	else:
		# Random displacement (original behavior with filter)
		eps = np.random.normal(0.0, std, size=(P.shape[0], 1))
		signs = np.random.choice([-1, 1], size=(P.shape[0], 1))
		eps = eps * signs
		P_displaced = P + eps * N
		mask = np.abs(h_gt_np(P_displaced)) > 0.05
		Q = P_displaced[mask]
		return Q

def sample_negative_spiral_3d(pos_pts, grad_gt_np, h_gt_np, std=0.15, proximity=0.5):
	
	"""Generate negative samples around positives in arbitrary directions.

	For each positive P we sample a displacement vector that is either along the
	local normal `N` or a random orthogonal-ish direction. The parameter
	`random_frac` controls the fraction of samples that use a random direction
	(0.0 = all normals, 1.0 = all random directions). This ensures negatives
	can appear in any direction around the curve.
	"""
	P = pos_pts
	N = grad_gt_np(P)
	n = P.shape[0]

	# which points use random directions vs normal-based
	# This ensures negatives are sampled in random orthogonal-ish directions for every positive.
	use_random = np.ones((n, 1), dtype=bool)

	# random directions (roughly isotropic): sample normal and orthonormalize
	rand_dirs = np.random.normal(size=(n, 3))
	# make them roughly orthogonal to N to avoid pure normal duplication
	rand_dirs = rand_dirs - (np.sum(rand_dirs * N, axis=1, keepdims=True) * N)
	rand_dirs /= (np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12)

	# final direction per point
	dirs = np.where(use_random, rand_dirs, N)

	# magnitudes (absolute) with random sign (in/out)
	mags = np.abs(np.random.normal(0.0, std, size=(n, 1)))
	# proximity scales the magnitude so values <1 place negatives closer to GT
	mags = mags * float(proximity)

	P_displaced = P + mags * dirs
	mask = np.abs(h_gt_np(P_displaced)) > 1e-2
	Q = P_displaced[mask]
	
	return Q

def sample_dense_negative_spiral_3d(pos_pts, h_gt_np, density_factor=50, min_dist=0.7):
	"""
	Generate much denser negatives for spiral: random points in the cube, filtered to be at least min_dist away from the curve.
	"""
	n_neg = len(pos_pts) if pos_pts is not None else 1000
	cube_lim = 2.5
	candidates = np.random.uniform(-cube_lim, cube_lim, size=(n_neg * density_factor, 3))
	dists = h_gt_np(candidates)
	mask = dists > min_dist
	negatives = candidates[mask]
	if len(negatives) > n_neg:
		negatives = negatives[:n_neg]
	return negatives

def sample_negative_spiral_tube(h_gt_np, n_samples=10000, 
                                 R0=1.0, alpha=0.3, r_tube=0.15,
                                 t_min=-2*np.pi, t_max=2*np.pi, 
                                 offset_radius=0.1):
	"""
	Generate negative samples for spiral TUBE constraint.
	Strategy: Sample uniformly in a bounding cylinder around the spiral,
	then filter to keep only points that are away from the tube surface.
	
	Args:
		h_gt_np: Ground truth implicit function (distance to tube surface)
		n_samples: Target number of negative samples
		R0: Base radius of spiral
		alpha: Pitch of spiral
		r_tube: Tube radius
		t_min, t_max: Parameter range for spiral
		offset_radius: Extra margin for cylinder bounds
		
	Returns:
		np.ndarray: Negative samples (N, 3)
	"""
	# Bounding cylinder: radius = R0 + r_tube + offset, height from z_min to z_max
	cylinder_radius = R0 + r_tube + offset_radius
	z_min = alpha * t_min
	z_max = alpha * t_max
	
	# Oversample to compensate for filtering
	n_candidates = int(n_samples * 3)
	# Deterministic seeding for reproducible negative sampling
	seed = 42
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	# Sample uniformly in cylinder
	theta = np.random.uniform(0, 2*np.pi, n_candidates)
	r = np.sqrt(np.random.uniform(0, cylinder_radius**2, n_candidates))
	z = np.random.uniform(z_min, z_max, n_candidates)
	
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	candidates = np.column_stack([x, y, z])
	
	# Filter: keep points with distance > threshold from tube surface
	# We want negatives both INSIDE and OUTSIDE the tube
	dists = np.abs(h_gt_np(candidates))
	threshold = 0.1  # Minimum distance from surface
	mask = dists > threshold
	
	negatives = candidates[mask][:n_samples]
	
	return negatives
	
def split_train_test_3d(pos_np, neg_np, eval_split=0.2, noise_level=0.0):
	pos = torch.tensor(pos_np, dtype=torch.float32)
	neg = torch.tensor(neg_np, dtype=torch.float32)
	train_size = int(pos.shape[0] * (1 - eval_split))
	pos_train = pos[:train_size]
	pos_eval = pos[train_size:]
	neg_train = neg[:train_size]
	neg_eval = neg[train_size:]
	if noise_level > 0.0:
		pos_train = pos_train + np.random.normal(0.0, noise_level, size=pos_train.shape)
		pos_train = torch.tensor(pos_train, dtype=torch.float32)
	return pos_train, pos_eval, neg_train, neg_eval

def generate_surface_trajectory(total_length, num_steps, h, grad, get_samples, GRID_LIM=2.5,
				newton_iters=50, newton_tol=1e-6, max_retries=3, seed=None, verbose=False):
	"""
	Generate a trajectory on the surface by walking in tangent direction.
	If the trajectory hits the boundary, it continues from the initial point
	in the opposite direction.

	Args:
		total_length: Total arc length to traverse
		num_steps: Number of steps
		h: Implicit function (signed distance)
		grad: Gradient function (normal direction)
		get_samples: Function to sample initial point on surface
		GRID_LIM: Spatial boundary limit

	Returns:
		np.ndarray: Array of points along the trajectory
	"""

	# optionally seed RNGs for reproducible trajectory generation
	if seed is not None:
		
		set_seed(int(seed))
		

	# precompute step size
	step_size = float(total_length) / float(num_steps)

	# Try several initial samples if the tangent walk fails to produce a curve
	# We'll adapt the effective step size on subsequent initial attempts to
	# improve chances on high-curvature regions.
	initial_retries = 8
	trajectory_points = None

	for init_attempt in range(initial_retries):
		# reduce effective per-attempt step size if earlier attempts failed
		# but cap the reduction so we don't produce extremely tiny local walks
		min_factor = 0.35
		eff_step = step_size * max(min_factor, (0.5 ** init_attempt))
		if verbose:
			print(f"generate_surface_trajectory: init_attempt={init_attempt}, eff_step={eff_step:.6g}")
		# Initial point on surface
		P_initial = get_samples(1)[0]
		# If the sampler provides a specialized trajectory generator (ellipsoid), use it
		if hasattr(get_samples, 'sample_trajectory'):
			if verbose:
				print('Using specialized sample_trajectory from gt_samples')
			# try to pass seed through if sampler accepts it
			try:
				return get_samples.sample_trajectory(total_length, num_steps, seed=seed)
			except TypeError:
				return get_samples.sample_trajectory(total_length, num_steps)
		# Compute initial tangent direction: choose a random direction
		# roughly orthogonal to the surface normal to produce diverse trajectories.
		# Use a local RNG seeded by `seed` (if provided) to make this reproducible.
		N_initial = grad(P_initial.reshape(1, -1))[0]
		if seed is not None:
			try:
				rng_tan = np.random.default_rng(int(seed))
			except Exception:
				rng_tan = np.random
		else:
			# preserve legacy behavior when no seed provided
			rng_tan = np.random

		# sample a random vector and make it orthogonal to N_initial
		def _random_tangent(normal, rng_local):
			for _ in range(10):
				r = rng_local.normal(size=3)
				# remove component along normal
				r = r - np.dot(r, normal) * normal
				nrm = np.linalg.norm(r)
				if nrm > 1e-6:
					return r / nrm
			# fallback deterministic tangent if random attempts fail
			z_axis = np.array([0.0, 0.0, 1.0])
			t_tmp = np.cross(normal, z_axis)
			if np.linalg.norm(t_tmp) < 1e-6:
				x_axis = np.array([1.0, 0.0, 0.0])
				t_tmp = np.cross(normal, x_axis)
			return t_tmp / (np.linalg.norm(t_tmp) + 1e-9)

		T_initial = _random_tangent(N_initial, rng_tan)

		# Build forward/backward walk for this initial point
		traj_pts = []
		traj_pts.append(P_initial)

		# First pass: forward direction from initial point
		P_current = P_initial
		T_current = T_initial

		for i in range(num_steps):
			# Try to take a step along the current tangent, with adaptive retries
			success = False
			trial_step = eff_step
			for attempt in range(max_retries):
				P_guess = P_current + T_current * trial_step

				# Newton projection with configurable iterations/tolerance
				P_corrected = P_guess.copy()
				for _ in range(newton_iters):
					h_val = float(h(P_corrected.reshape(1, -1))[0])
					if abs(h_val) < newton_tol:
						break
					N_corrected = grad(P_corrected.reshape(1, -1))[0]
					P_corrected = P_corrected - h_val * N_corrected

				# If projection did not converge sufficiently or went out of bounds, reduce trial step
				final_h = float(h(P_corrected.reshape(1, -1))[0])
				if (np.abs(P_corrected) <= GRID_LIM).all() and abs(final_h) <= max(1e-3, newton_tol * 10):
					success = True
					break
				else:
					trial_step *= 0.5
					# if trial_step becomes extremely small, consider this attempt failed
					if trial_step < 1e-6:
						if verbose:
							print(f"  step reduction too small (trial_step<{trial_step:.1e}), aborting forward pass")
						break

			if not success:
				# If we cannot project a forward step even after retries, stop the forward pass
				break

			# Update tangent: project current tangent to new tangent plane and re-normalize
			N_new = grad(P_corrected.reshape(1, -1))[0]
			T_next = T_current - np.dot(T_current, N_new) * N_new
			norm_T = np.linalg.norm(T_next)
			if norm_T < 1e-6:
				# Fallback: construct any tangent vector orthogonal to N_new
				x_axis = np.array([1.0, 0.0, 0.0])
				t_tmp = np.cross(N_new, x_axis)
				if np.linalg.norm(t_tmp) < 1e-6:
					y_axis = np.array([0.0, 1.0, 0.0])
					t_tmp = np.cross(N_new, y_axis)
				T_next = t_tmp / (np.linalg.norm(t_tmp) + 1e-9)
			else:
				T_next = T_next / (norm_T + 1e-9)

			# Keep tangent continuity (avoid flipping sign unexpectedly)
			if np.dot(T_next, T_current) < 0:
				T_next = -T_next

			traj_pts.append(P_corrected)
			P_current = P_corrected
			T_current = T_next

		# Second pass: backward direction from initial point (opposite direction)
		# Only if we stopped early (hit boundary) and haven't used all steps
		steps_used = len(traj_pts) - 1  # -1 because initial point is already counted
		remaining_steps = num_steps - steps_used

		if remaining_steps > 0:
			P_current = P_initial
			T_current = -T_initial  # Reverse direction

			backward_points = []
			for i in range(remaining_steps):
				# Try retries like forward pass
				success = False
				trial_step = eff_step
				for attempt in range(max_retries):
					P_guess = P_current + T_current * trial_step

					P_corrected = P_guess.copy()
					for _ in range(newton_iters):
						h_val = float(h(P_corrected.reshape(1, -1))[0])
						if abs(h_val) < newton_tol:
							break
						N_corrected = grad(P_corrected.reshape(1, -1))[0]
						P_corrected = P_corrected - h_val * N_corrected
					final_h = float(h(P_corrected.reshape(1, -1))[0])
					if (np.abs(P_corrected) <= GRID_LIM).all() and abs(final_h) <= max(1e-3, newton_tol * 10):
						success = True
						break
					else:
						trial_step *= 0.5
						if trial_step < 1e-6:
							if verbose:
								print(f"  backward: step reduction too small (trial_step<{trial_step:.1e}), aborting backward pass")
							break

				if not success:
					break

				N_new = grad(P_corrected.reshape(1, -1))[0]
				T_next = T_current - np.dot(T_current, N_new) * N_new
				norm_T = np.linalg.norm(T_next)
				if norm_T < 1e-6:
					x_axis = np.array([1.0, 0.0, 0.0])
					t_tmp = np.cross(N_new, x_axis)
					if np.linalg.norm(t_tmp) < 1e-6:
						y_axis = np.array([0.0, 1.0, 0.0])
						t_tmp = np.cross(N_new, y_axis)
					T_next = t_tmp / (np.linalg.norm(t_tmp) + 1e-9)
				else:
					T_next = T_next / (norm_T + 1e-9)

				if np.dot(T_next, T_current) < 0:
					T_next = -T_next

				backward_points.append(P_corrected)
				P_current = P_corrected
				T_current = T_next

			# Combine: backward points (reversed) + initial point + forward points
			if backward_points:
				backward_points.reverse()
				traj_pts = backward_points + traj_pts

		# If this attempt produced at least 2 trajectory points, accept it
		if len(traj_pts) >= 2:
			trajectory_points = traj_pts
			if verbose:
				print(f"  init_attempt {init_attempt} succeeded with {len(traj_pts)} points (eff_step={eff_step:.6g})")
			break
		else:
			if verbose:
				print(f"  init_attempt {init_attempt} failed produced {len(traj_pts)} pts (eff_step={eff_step:.6g})")

	# If after retries we still have too few points, fall back to sampling but warn
	if trajectory_points is None or len(trajectory_points) < 2:
		print('Warning: generate_surface_trajectory failed to build a continuous walk; falling back to sampled surface points')
		traj = np.atleast_2d(get_samples(max(num_steps, 1)))
		projected = []
		for p in traj:
			Pp = p.copy()
			for _ in range(max(10, newton_iters)):
				h_val = float(h(Pp.reshape(1, -1))[0])
				if abs(h_val) < newton_tol:
					break
				Np = grad(Pp.reshape(1, -1))[0]
				Pp = Pp - h_val * Np
			projected.append(Pp)
		projected = np.array(projected)
		# pad if necessary
		while projected.shape[0] < num_steps:
			projected = np.vstack([projected, projected[-1]])
		return projected[:num_steps]

	# continue with existing resampling/projection flow using trajectory_points
	traj = np.array(trajectory_points)

	# Ensure output is exactly `num_steps` points: if we have fewer, resample
	traj = np.array(trajectory_points)
	if traj.shape[0] < 2:
		# Fallback: sample points from get_samples and project them
		pts = np.atleast_2d(get_samples(max(num_steps, 1)))
		projected = []
		for i, p in enumerate(pts):
			Pp = p.copy()
			for _ in range(max(10, newton_iters)):
				h_val = float(h(Pp.reshape(1, -1))[0])
				if abs(h_val) < newton_tol:
					break
				Np = grad(Pp.reshape(1, -1))[0]
				Pp = Pp - h_val * Np
			projected.append(Pp)
		projected = np.array(projected)
		if projected.shape[0] >= num_steps:
			return projected[:num_steps]
		# If still short, repeat last
		while projected.shape[0] < num_steps:
			projected = np.vstack([projected, projected[-1]])
		return projected

	# If we have at least two points, resample along arc-length to exactly num_steps
	# Compute cumulative arc-length
	diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
	cum = np.concatenate([[0.0], np.cumsum(diffs)])
	total_used = cum[-1] if cum[-1] > 0 else 1.0
	s_new = np.linspace(0.0, total_used, num_steps)
	x_new = np.interp(s_new, cum, traj[:, 0])
	y_new = np.interp(s_new, cum, traj[:, 1])
	z_new = np.interp(s_new, cum, traj[:, 2])
	interp_pts = np.stack([x_new, y_new, z_new], axis=1)

	# Project interpolated points back onto the surface for better adherence
	projected = []
	for P in interp_pts:
		Pp = P.copy()
		for _ in range(max(10, newton_iters * 2)):
			h_val = float(h(Pp.reshape(1, -1))[0])
			if abs(h_val) < newton_tol:
				break
			Np = grad(Pp.reshape(1, -1))[0]
			Pp = Pp - h_val * Np
		# Clamp to grid limits if needed
		if not (np.abs(Pp) <= GRID_LIM).all():
			Pp = np.clip(Pp, -GRID_LIM, GRID_LIM)
		projected.append(Pp)

	return np.array(projected)