"""
Utilities for generating trajectories on spherical surfaces.
Includes SLERP interpolation and sinusoidal perturbations.
"""

import numpy as np


def slerp(p0, p1, t):
    """Spherical linear interpolation (SLERP) between two points.
    
    Args:
        p0: Start point (3,)
        p1: End point (3,)
        t: Interpolation parameter [0, 1]
        
    Returns:
        np.ndarray: Interpolated point (3,)
    """
    p0n = p0 / np.linalg.norm(p0)
    p1n = p1 / np.linalg.norm(p1)
    omega = np.arccos(np.clip(np.dot(p0n, p1n), -1, 1))
    
    # Handle degenerate case
    if omega < 1e-8:
        return p0n * np.linalg.norm(p0)
    
    sin_omega = np.sin(omega)
    R = np.linalg.norm(p0)
    return R * (np.sin((1 - t) * omega) * p0n + np.sin(t * omega) * p1n) / sin_omega


def perturb_point_on_sphere(p, amplitude, freq, t, R):
    """Add sinusoidal perturbation to point on sphere while staying on surface.
    
    Args:
        p: Base point on sphere (3,)
        amplitude: Amplitude of oscillation
        freq: Frequency of oscillation
        t: Time parameter [0, 1]
        R: Sphere radius
        
    Returns:
        np.ndarray: Perturbed point on sphere (3,)
    """
    # Compute tangent direction (perpendicular to radial)
    tangent = np.cross(p, np.array([0, 0, 1]))
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.cross(p, np.array([0, 1, 0]))
    tangent = tangent / np.linalg.norm(tangent)
    
    # Add sinusoidal perturbation
    perturbed = p + amplitude * np.sin(freq * t * 2 * np.pi) * tangent
    
    # Project back onto sphere
    return R * perturbed / np.linalg.norm(perturbed)

def generate_sinusoidal_reference(start, goal, R, N, amplitude=None, freq=3):
    """
    Genera una trajectòria sinusoidal PLANAR en un pla VERTICAL.
    
    La trajectòria és una línia recta d'start a goal (projecció XY) amb oscil·lacions
    sinusoidals en l'eix Z (vertical).
    L'MPC ha de seguir aquesta referència planar però mantenint-se a l'esfera.
    """
    if amplitude is None:
        amplitude = 0.3 * R
    
    # Generar trajectòria sinusoidal en un pla vertical
    ref_planar = []
    
    # Vector direcció de start a goal (només en el pla XY)
    start_2d = start[:2]  # Només x, y
    goal_2d = goal[:2]
    direction = goal_2d - start_2d
    dist = np.linalg.norm(direction)
    
    if dist < 1e-8:
        # Si start i goal són molt a prop en XY, crear línia recta en X
        direction_normalized = np.array([1.0, 0.0])
    else:
        direction_normalized = direction / dist
    
    for k in range(N):
        t = k / max(N - 1, 1)
        
        # Punt base sobre la línia recta en el pla XY
        base_2d = start_2d + t * direction
        
        # Oscil·lació en l'eix Z (vertical)
        z_oscillation = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Punt final amb oscil·lació vertical
        pt_planar = np.array([base_2d[0], base_2d[1], z_oscillation])
        
        ref_planar.append(pt_planar)
    
    # Retornar trajectòria PLANAR (sense projectar a esfera)
    return np.array(ref_planar)


def random_point_on_sphere(R, seed=None):
    """Generate random point uniformly distributed on sphere.
    
    Args:
        R: Sphere radius
        seed: Random seed for reproducibility
        
    Returns:
        np.ndarray: Point on sphere (3,)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(3)
    else:
        v = np.random.randn(3)
    
    v = v / np.linalg.norm(v)
    return R * v


__all__ = [
    'slerp',
    'perturb_point_on_sphere', 
    'generate_sinusoidal_reference',
    'random_point_on_sphere'
]
