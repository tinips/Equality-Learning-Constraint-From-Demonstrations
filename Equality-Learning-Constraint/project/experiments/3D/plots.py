import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# MPC Basic Animation
np.random.seed(42)

# Paràmetres del sistema
dt = 0.1
N_horizon = 10  # Horitzó de predicció
T_total = 50    # Passos totals de simulació

# Trajectòria de referència (cercle)
t_ref = np.linspace(0, 2*np.pi, 100)
x_ref = 5 + 3*np.cos(t_ref)
y_ref = 5 + 3*np.sin(t_ref)

# Estats inicials
x_traj = [2.0]
y_traj = [5.0]

# Simulació del moviment (simple seguiment)
for i in range(T_total):
    # Trobar punt més proper a la referència
    idx = int((i / T_total) * len(x_ref))
    target_x = x_ref[idx % len(x_ref)]
    target_y = y_ref[idx % len(y_ref)]
    
    # Control simple cap al target
    dx = 0.15 * (target_x - x_traj[-1])
    dy = 0.15 * (target_y - y_traj[-1])
    
    x_traj.append(x_traj[-1] + dx)
    y_traj.append(y_traj[-1] + dy)

# Setup de la figura
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('X Position', fontsize=12)
ax.set_ylabel('Y Position', fontsize=12)
ax.set_title('MPC Basic Control', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Elements del plot
ref_line, = ax.plot(x_ref, y_ref, 'g--', linewidth=2, alpha=0.5, label='Reference Trajectory')
actual_line, = ax.plot([], [], 'b-', linewidth=2, label='Actual Trajectory')
current_pos, = ax.plot([], [], 'ro', markersize=12, label='Current Position')
predicted_line, = ax.plot([], [], 'm--', linewidth=2, alpha=0.7, label='Predicted Horizon')
predicted_points, = ax.plot([], [], 'mo', markersize=6, alpha=0.7)

ax.legend(loc='upper right', fontsize=10)

def init():
    actual_line.set_data([], [])
    current_pos.set_data([], [])
    predicted_line.set_data([], [])
    predicted_points.set_data([], [])
    return actual_line, current_pos, predicted_line, predicted_points

def animate(frame):
    # Trajectòria actual fins ara
    actual_line.set_data(x_traj[:frame+1], y_traj[:frame+1])
    
    # Posició actual
    current_pos.set_data([x_traj[frame]], [y_traj[frame]])
    
    # Horitzó de predicció (simulació simple)
    if frame + N_horizon < len(x_traj):
        pred_x = x_traj[frame:frame+N_horizon]
        pred_y = y_traj[frame:frame+N_horizon]
        predicted_line.set_data(pred_x, pred_y)
        predicted_points.set_data(pred_x, pred_y)
    else:
        # Predicció lineal simple per als últims frames
        remaining = N_horizon - (len(x_traj) - frame)
        pred_x = x_traj[frame:] + [x_traj[-1]] * max(0, remaining)
        pred_y = y_traj[frame:] + [y_traj[-1]] * max(0, remaining)
        predicted_line.set_data(pred_x, pred_y)
        predicted_points.set_data(pred_x, pred_y)
    
    return actual_line, current_pos, predicted_line, predicted_points

# Crear animació
anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_traj)-1, 
                     interval=100, blit=True, repeat=True)

plt.tight_layout()
plt.show()
