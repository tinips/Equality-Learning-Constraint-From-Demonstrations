import numpy as np
import torch

def make_constraint(name: str, grid_lim=1.8):

	name = name.lower()
	if name == "circle":
		def h_np(P):
			x, y = P[...,0], P[...,1]
			return x*x + y*y - 1.0
		def grad_np(P):
			x, y = P[...,0], P[...,1]
			g = np.stack([2*x, 2*y], axis=-1)
			n = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
			return g / n
		def gt_samples(n=2000):
			t = np.random.uniform(0, 2*np.pi, n)
			return np.stack([np.cos(t), np.sin(t)], axis=1)
		def gt_curve(M=800):
			t = np.linspace(0, 2*np.pi, M, endpoint=False)
			return np.stack([np.cos(t), np.sin(t)], axis=1)
	elif name == "ellipse":
		a, b, theta = 1.2, 0.7, np.deg2rad(30.0)
		R = np.array([[np.cos(theta), -np.sin(theta)],
					  [np.sin(theta),  np.cos(theta)]])
		R_inv = R.T
		def h_np(P):
			Q = P @ R_inv.T
			X, Y = Q[...,0], Q[...,1]
			return (X/a)**2 + (Y/b)**2 - 1.0
		def grad_np(P):
			Q = P @ R_inv.T
			X, Y = Q[...,0], Q[...,1]
			gQ = np.stack([2*X/(a*a), 2*Y/(b*b)], axis=-1)
			g = gQ @ R_inv
			n = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
			return g / n
		def gt_samples(n=2000):
			t = np.random.uniform(0, 2*np.pi, n)
			Q = np.stack([a*np.cos(t), b*np.sin(t)], axis=1)
			return Q @ R.T
		def gt_curve(M=800):
			t = np.linspace(0, 2*np.pi, M, endpoint=False)
			Q = np.stack([a*np.cos(t), b*np.sin(t)], axis=1)
			return Q @ R.T
	elif name == "line":
		def h_np(P):
			x, y = P[...,0], P[...,1]
			return x + y - 1.0
		def grad_np(P):
			g = np.array([1.0,1.0], dtype=float)
			g = np.broadcast_to(g, P.shape)
			n = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
			return g / n
		def gt_samples(n=2000):
			xs = np.random.uniform(-grid_lim, grid_lim, n)
			ys = 1.0 - xs
			mask = (np.abs(ys) <= grid_lim)
			xs, ys = xs[mask], ys[mask]
			return np.stack([xs, ys], axis=1)
		def gt_curve(M=800):
			xs = np.linspace(-grid_lim, grid_lim, M)
			ys = 1.0 - xs
			mask = (np.abs(ys) <= grid_lim)
			return np.stack([xs[mask], ys[mask]], axis=1)
	elif name == "parabola":
		def h_np(P):
			x, y = P[...,0], P[...,1]
			return y - (0.5*x*x - 0.2)
		def grad_np(P):
			x, y = P[...,0], P[...,1]
			g = np.stack([-x, np.ones_like(x)], axis=-1)
			n = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
			return g / n
		def gt_samples(n=2000):
			xs = np.random.uniform(-1.6, 1.6, n)
			ys = 0.5*xs*xs - 0.2
			mask = (np.abs(ys) <= grid_lim)
			xs, ys = xs[mask], ys[mask]
			return np.stack([xs, ys], axis=1)
		def gt_curve(M=800):
			xs = np.linspace(-1.6, 1.6, M)
			ys = 0.5*xs*xs - 0.2
			mask = (np.abs(ys) <= grid_lim)
			return np.stack([xs[mask], ys[mask]], axis=1)
	elif name == "hiperbola":
		a, b, theta = 1.0, 0.5, np.deg2rad(20.0)
		R = np.array([[np.cos(theta), -np.sin(theta)],
					  [np.sin(theta),  np.cos(theta)]])
		R_inv = R.T
		def h_np(P):
			Q = P @ R_inv.T
			X, Y = Q[...,0], Q[...,1]
			return (X/a)**2 - (Y/b)**2 - 1.0
		def grad_np(P):
			Q = P @ R_inv.T
			X, Y = Q[...,0], Q[...,1]
			gQ = np.stack([2*X/(a*a), -2*Y/(b*b)], axis=-1)
			g  = gQ @ R_inv
			n  = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
			return g / n
		def _branch_samples(t):
			Xp =  a * np.cosh(t)
			Yp =  b * np.sinh(t)
			Xn = -a * np.cosh(t)
			Yn =  b * np.sinh(t)
			Q  = np.concatenate([np.stack([Xp, Yp], axis=1),
								np.stack([Xn, Yn], axis=1)], axis=0)
			return Q @ R.T
		def gt_samples(n=2000):
			t = np.random.uniform(-2.0, 2.0, size=(n//2,))
			P = _branch_samples(t)
			mask = (np.abs(P[:,0]) <= grid_lim) & (np.abs(P[:,1]) <= grid_lim)
			P = P[mask]
			if P.shape[0] < n:
				extra = _branch_samples(np.random.uniform(-2.0, 2.0, size=((n - P.shape[0])//2 + 1,)))
				P = np.concatenate([P, extra], axis=0)
				mask = (np.abs(P[:,0]) <= grid_lim) & (np.abs(P[:,1]) <= grid_lim)
				P = P[mask]
			idx = np.random.permutation(min(n, P.shape[0]))
			return P[idx[:n]]
		def gt_curve(M=800):
			t = np.linspace(-2.0, 2.0, M//2, endpoint=True)
			P = _branch_samples(t)
			mask = (np.abs(P[:,0]) <= grid_lim) & (np.abs(P[:,1]) <= grid_lim)
			return P[mask]
	elif name == "lemniscate":
		a = 1.0
		def h_np(P):
			x, y = P[...,0], P[...,1]
			S = x*x + y*y
			return (S*S) - (a*a)*(x*x - y*y)
		def grad_np(P):
			x, y = P[...,0], P[...,1]
			S = x*x + y*y
			gx = 4*x*S - 2*(a*a)*x
			gy = 4*y*S + 2*(a*a)*y
			g = np.stack([gx, gy], axis=-1)
			n = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-9
			return g / n
		def gt_samples(n=2000):
			center_extra_ratio = 0.5 
			
			n_normal = int(n * (1 - center_extra_ratio))
			theta_normal = np.random.uniform(0, 2*np.pi, n_normal*2)
			c2_normal = np.cos(2*theta_normal)
			mask_normal = c2_normal >= 0
			theta_normal = theta_normal[mask_normal]
			c2_normal = c2_normal[mask_normal]
			
			if len(theta_normal) >= n_normal:
				theta_normal = theta_normal[:n_normal]
				c2_normal = c2_normal[:n_normal]
			
			r_normal = a * np.sqrt(c2_normal)
			x_normal = r_normal * np.cos(theta_normal)
			y_normal = r_normal * np.sin(theta_normal)
			
			n_center = n - len(theta_normal)
			if n_center > 0:
				transition_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
				center_angles = []
				
				for base_angle in transition_angles:
					n_per_transition = n_center // 4
					if n_per_transition > 0:
						angle_spread = 0.3  
						angles_around = np.random.uniform(
							base_angle - angle_spread, 
							base_angle + angle_spread, 
							n_per_transition
						)
						center_angles.extend(angles_around)
				
				remaining = n_center - len(center_angles)
				if remaining > 0:
					extra_angles = np.random.uniform(0, 2*np.pi, remaining)
					center_angles.extend(extra_angles)
				
				center_angles = np.array(center_angles[:n_center])
				c2_center = np.cos(2*center_angles)
				
				valid_mask = c2_center >= 0
				center_angles = center_angles[valid_mask]
				c2_center = c2_center[valid_mask]
				
				if len(center_angles) > 0:
					r_center = a * np.sqrt(c2_center)
					x_center = r_center * np.cos(center_angles)
					y_center = r_center * np.sin(center_angles)
					
					x = np.concatenate([x_normal, x_center])
					y = np.concatenate([y_normal, y_center])
				else:
					x, y = x_normal, y_normal
			else:
				x, y = x_normal, y_normal
			
			P = np.stack([x, y], axis=1)
			
			m = (np.abs(P[:,0]) <= grid_lim) & (np.abs(P[:,1]) <= grid_lim)
			P = P[m]
			
			if P.shape[0] > n:
				idx = np.random.permutation(P.shape[0])[:n]
				P = P[idx]
			elif P.shape[0] < n:
				extra_needed = n - P.shape[0]
				theta_extra = np.random.uniform(0, 2*np.pi, extra_needed*3)
				c2_extra = np.cos(2*theta_extra)
				mask_extra = c2_extra >= 0
				theta_extra = theta_extra[mask_extra][:extra_needed]
				c2_extra = c2_extra[mask_extra][:extra_needed]
				r_extra = a * np.sqrt(c2_extra)
				x_extra = r_extra * np.cos(theta_extra)
				y_extra = r_extra * np.sin(theta_extra)
				P_extra = np.stack([x_extra, y_extra], axis=1)
				P = np.concatenate([P, P_extra], axis=0)
			
			return P[:n]
		def gt_curve(M=1600):
			theta = np.linspace(0, 2*np.pi, M*2, endpoint=False)
			c2 = np.cos(2*theta)
			mask = c2 >= 0
			theta = theta[mask]
			c2 = c2[mask]
			r = a * np.sqrt(c2)
			x = r * np.cos(theta)
			y = r * np.sin(theta)
			P = np.stack([x, y], axis=1)
			m = (np.abs(P[:,0]) <= grid_lim) & (np.abs(P[:,1]) <= grid_lim)
			return P[m]
	
		
	else:
		raise ValueError(f"Unknown constraint: {name}")
	return h_np, grad_np, gt_samples, gt_curve

def sample_positive(gt_samples_fn, n):
	return gt_samples_fn(n)

def sample_negative_from_normals(pos_pts, grad_gt_np, h_gt_np, std=0.15):
	P = pos_pts
	N = grad_gt_np(P)
	eps = np.random.normal(0.0, std, size=(P.shape[0], 1))
	P_displaced = P + eps * N
	mask = np.abs(h_gt_np(P_displaced)) > 1e-3
	Q = P_displaced[mask]
	return Q

def split_train_test(pos_np, neg_np, eval_split=0.2, noise_level=0.0):
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
