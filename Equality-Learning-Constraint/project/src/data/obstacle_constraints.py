"""
Obstacle constraint functions for MPC inequality constraints.
These define forbidden regions (h >= 0 to stay outside obstacle).
"""

import numpy as np


def make_cylinder_obstacle(radius=1, axis='z', center=None):
    """
    Create a cylindrical obstacle (infinite along axis direction).
    
    Args:
        radius: Cylinder radius
        axis: Axis direction ('x', 'y', or 'z')
        center: Center point in the plane perpendicular to axis (2D)
                Default: [0, 0] for the perpendicular plane
    
    Returns:
        h_obstacle: Function h(x) where h >= 0 means outside (safe), h < 0 means inside (forbidden)
        grad_obstacle: Gradient function
    """
    if center is None:
        center = np.array([0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)
    
    axis = axis.lower()
    
    if axis == 'z':
        # Cylinder along Z axis: forbidden region where sqrt(x^2 + y^2) < radius
        def h_obstacle(P):
            P = np.atleast_2d(np.asarray(P, dtype=np.float64))
            x = P[:, 0] - center[0]
            y = P[:, 1] - center[1]
            r_xy = np.sqrt(x * x + y * y)
            # h >= 0 means outside (safe), h < 0 means inside (forbidden)
            return r_xy - radius
        
        def grad_obstacle(P):
            P = np.atleast_2d(np.asarray(P, dtype=np.float64))
            x = P[:, 0] - center[0]
            y = P[:, 1] - center[1]
            r_xy = np.sqrt(x * x + y * y) + 1e-12
            grads = np.zeros_like(P)
            grads[:, 0] = x / r_xy
            grads[:, 1] = y / r_xy
            grads[:, 2] = 0.0
            return grads if P.shape[0] > 1 else grads[0]
    
    elif axis == 'y':
        # Cylinder along Y axis: forbidden region where sqrt(x^2 + z^2) < radius
        def h_obstacle(P):
            P = np.atleast_2d(np.asarray(P, dtype=np.float64))
            x = P[:, 0] - center[0]
            z = P[:, 2] - center[1]
            r_xz = np.sqrt(x * x + z * z)
            return r_xz - radius
        
        def grad_obstacle(P):
            P = np.atleast_2d(np.asarray(P, dtype=np.float64))
            x = P[:, 0] - center[0]
            z = P[:, 2] - center[1]
            r_xz = np.sqrt(x * x + z * z) + 1e-12
            grads = np.zeros_like(P)
            grads[:, 0] = x / r_xz
            grads[:, 1] = 0.0
            grads[:, 2] = z / r_xz
            return grads if P.shape[0] > 1 else grads[0]
    
    elif axis == 'x':
        # Cylinder along X axis: forbidden region where sqrt(y^2 + z^2) < radius
        def h_obstacle(P):
            P = np.atleast_2d(np.asarray(P, dtype=np.float64))
            y = P[:, 1] - center[0]
            z = P[:, 2] - center[1]
            r_yz = np.sqrt(y * y + z * z)
            return r_yz - radius
        
        def grad_obstacle(P):
            P = np.atleast_2d(np.asarray(P, dtype=np.float64))
            y = P[:, 1] - center[0]
            z = P[:, 2] - center[1]
            r_yz = np.sqrt(y * y + z * z) + 1e-12
            grads = np.zeros_like(P)
            grads[:, 0] = 0.0
            grads[:, 1] = y / r_yz
            grads[:, 2] = z / r_yz
            return grads if P.shape[0] > 1 else grads[0]
    
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
    
    return h_obstacle, grad_obstacle


def make_sphere_obstacle(radius=0.5, center=None):
    """
    Create a spherical obstacle.
    
    Args:
        radius: Sphere radius
        center: Sphere center (3D point)
    
    Returns:
        h_obstacle: Function h(x) where h >= 0 means outside (safe)
        grad_obstacle: Gradient function
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)
    
    def h_obstacle(P):
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        d = P - center.reshape(1, -1)
        r = np.linalg.norm(d, axis=1)
        return r - radius
    
    def grad_obstacle(P):
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        d = P - center.reshape(1, -1)
        r = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        grads = d / r
        return grads if P.shape[0] > 1 else grads[0]
    
    return h_obstacle, grad_obstacle


def make_rectangular_box_obstacle(x_half=1, y_half=0.3, z_half=0.5, center=None):
    """
    Create a rectangular box obstacle centered at 'center'.
    Forbids region inside the box.
    
    Args:
        x_half: Half-width in x direction
        y_half: Half-width in y direction
        z_half: Half-height in z direction
        center: Center of box [x, y, z], default [0, 0, 0]
    
    Returns:
        h_obstacle: Function h(x) where h >= 0 means outside (safe), h < 0 means inside (forbidden)
        grad_obstacle: Gradient function
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center, dtype=np.float64)
    
    def h_obstacle(P):
        """Distance to box surface (negative inside, positive outside)."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        
        # Translate to box-centered coordinates
        rel = P - center.reshape(1, -1)
        
        # Distance to each face (negative if beyond that face inward)
        dx = np.abs(rel[:, 0]) - x_half
        dy = np.abs(rel[:, 1]) - y_half
        dz = np.abs(rel[:, 2]) - z_half
        
        # Distance to box: positive outside, negative inside
        # If all d_i < 0, we're inside (use max of the three)
        # If any d_i >= 0, we're outside (use Euclidean distance to corner/edge)
        inside_mask = (dx < 0) & (dy < 0) & (dz < 0)
        
        # For inside points: distance is max(dx, dy, dz) (most negative -> closest to surface)
        dist_inside = np.maximum(np.maximum(dx, dy), dz)
        
        # For outside points: distance to nearest point on box
        dx_out = np.maximum(dx, 0)
        dy_out = np.maximum(dy, 0)
        dz_out = np.maximum(dz, 0)
        dist_outside = np.sqrt(dx_out**2 + dy_out**2 + dz_out**2)
        
        result = np.where(inside_mask, dist_inside, dist_outside)
        return result if P.shape[0] > 1 else result[0]
    
    def grad_obstacle(P):
        """Gradient of distance function."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        
        # Translate to box-centered coordinates
        rel = P - center.reshape(1, -1)
        
        dx = np.abs(rel[:, 0]) - x_half
        dy = np.abs(rel[:, 1]) - y_half
        dz = np.abs(rel[:, 2]) - z_half
        
        inside_mask = (dx < 0) & (dy < 0) & (dz < 0)
        
        grads = np.zeros_like(P)
        
        for i in range(P.shape[0]):
            if inside_mask[i]:
                # Inside: gradient points to nearest face
                if dx[i] >= dy[i] and dx[i] >= dz[i]:
                    grads[i, 0] = np.sign(rel[i, 0])
                elif dy[i] >= dz[i]:
                    grads[i, 1] = np.sign(rel[i, 1])
                else:
                    grads[i, 2] = np.sign(rel[i, 2])
            else:
                # Outside: gradient points to nearest point on box
                if dx[i] > 0:
                    grads[i, 0] = np.sign(rel[i, 0])
                if dy[i] > 0:
                    grads[i, 1] = np.sign(rel[i, 1])
                if dz[i] > 0:
                    grads[i, 2] = np.sign(rel[i, 2])
                    
                # Normalize if outside (corner/edge case)
                norm = np.linalg.norm(grads[i]) + 1e-12
                grads[i] = grads[i] / norm
        
        return grads if P.shape[0] > 1 else grads[0]
    
    return h_obstacle, grad_obstacle

