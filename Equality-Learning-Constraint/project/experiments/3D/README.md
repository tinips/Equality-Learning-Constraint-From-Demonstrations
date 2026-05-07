# 3D Implicit Surface Learning Experiments

Unified pipeline for training and evaluating implicit neural networks on 3D constraints.

## Quick Start

```bash
# Run ALL constraints (default if no --constraint specified)
python main_3d.py

# Run single constraint
python main_3d.py --constraint sphere

# Run multiple constraints at once
python main_3d.py --constraint sphere cylinder hyperplane

# Run spiral experiment with specific architecture
python main_3d.py --constraint spiral --archs NN

# Run multiple constraints with multiple architectures
python main_3d.py --constraint sphere cylinder --archs NN ParamQuad

# Run with multiple noise levels
python main_3d.py --constraint sphere --noise 0.01 0.03 0.05

# Use trajectory dataset instead of random samples
python main_3d.py --constraint hyperplane --dataset trajectory
python main_3d.py --constraint sphere cylinder --dataset trajectory --traj-num 10 --traj-length 5.0
```

## Dataset Types

### Standard Dataset (default)
Random samples uniformly distributed on the surface using the constraint's sampling function.
- Good for general surface learning
- Isotropic coverage of the surface

### Trajectory Dataset
Connected sequences of points forming smooth paths along the surface.
- Useful for learning temporal/sequential patterns
- Simulates motion along the constraint surface
- Parameters:
  - `--traj-num`: Number of separate trajectories (default: 5)
  - `--traj-length`: Total arc length of each trajectory (default: 4.0)
  - `--traj-points`: Number of points per trajectory (default: 49)

## Available Architectures

| Architecture | Description | Default LR |
|-------------|-------------|-----------|
| `NN` | Neural network (SpiralImplicitMLP) | config.LR_NN |
| `ParamQuad` | Parametric quadratic | config.LR_POLY |
| `ParamLinear` | Parametric linear | config.LR_POLY |

## Output Structure

Results are saved to `experiments/3D/{constraint}/outputs/noise_{level}_{arch}[_trajectory]/`:

```
sphere/
  outputs/
    noise_0.03_NN/              # Standard dataset, NN architecture
      figB_curves.png
      figD_heatmap.png
      figE_confusion_matrices.png
      figI_GT_samples.png
      figI_contours_*.png
      figJ_projection_NN.png
      slices/
        slices_NN_Margin_Loss.png
        slices_NN_KNN_Loss.png
        slices_NN_Neural_Pull_Loss.png
    noise_0.03_NN_trajectory/   # Trajectory dataset, NN architecture
      ... (same structure)
    noise_0.03_ParamQuad/       # Standard dataset, ParamQuad architecture
      ...
```

**Note:** Each combination of (noise, architecture, dataset type) gets its own subfolder!

## Loss Functions

The pipeline trains models with three loss variants:

1. **Margin Loss** (variant 2): Hinge loss with margin separation
2. **KNN Loss** (variant 5): K-nearest neighbors consistency
3. **Neural Pull Loss** (variant 6): Gradient-based projection

## Advanced Options

### Run All Constraints
```bash
# Train on all 4 constraints with same settings
python main_3d.py --constraint sphere spiral cylinder hyperplane --archs NN

# Results will be organized in separate folders:
# - sphere/outputs/noise_0.03_NN/
# - spiral/outputs/noise_0.03_NN/
# - cylinder/outputs/noise_0.03_NN/
# - hyperplane/outputs/noise_0.03_NN/
```

### Multiple Architectures + Multiple Noise Levels
```bash
# Test NN and ParamQuad with 3 noise levels
python main_3d.py --constraint sphere --archs NN ParamQuad --noise 0.01 0.03 0.05

# Results will be saved to:
# - sphere/outputs/noise_0.01_NN/
# - sphere/outputs/noise_0.01_ParamQuad/
# - sphere/outputs/noise_0.03_NN/
# - sphere/outputs/noise_0.03_ParamQuad/
# - sphere/outputs/noise_0.05_NN/
# - sphere/outputs/noise_0.05_ParamQuad/
```

### Disable Specific Plots
```bash
# Skip time-consuming plots
python main_3d.py --constraint sphere --disable-plots projection slices

# Only generate essential plots
python main_3d.py --constraint spiral --disable-plots heatmap confusion contours
```

### Save Plots Without Displaying Them
```bash
# Useful for batch processing or remote execution
python main_3d.py --no-show

# Run all constraints in background without popup windows
python main_3d.py --constraint sphere cylinder hyperplane --no-show

# Combine with other options
python main_3d.py --archs NN ParamQuad --noise 0.01 0.03 --no-show
```

### Custom Architectures
```bash
# Use different architecture (when implemented)
python main_3d.py --constraint sphere --archs NN ParamQuad
```

## Slice Visualization

The 2D slice visualization uses **automatic intelligent view selection**:

- **Zero-crossing detection**: Finds where surfaces intersect slice planes
- **K-means clustering**: Selects 5 most representative regions
- **Adaptive zoom**: Window size adjusts to surface complexity

Each visualization shows:
- **Row 1**: XY plane at `z_levels[0]`
- **Row 2**: XY plane at `z_levels[1]`
- **Row 3**: XZ plane at `y_level`

For **spiral** (1D curve):
- GT shown as **red scatter points** (⚫) where curve intersects plane
- Learned surface shown as **blue contours**

For **surfaces** (sphere, cylinder, hyperplane):
- GT shown as **red contours**
- Learned surface shown as **blue dashed contours**

## Configuration

Edit `CONSTRAINT_CONFIGS` in `main_3d.py` to customize:

```python
CONSTRAINT_CONFIGS = {
    "sphere": {
        "archs": ["NN"],
        "z_levels": (0.0, 0.5),   # Z values for XY slices
        "y_level": 0.0,            # Y value for XZ slice
        "bounds": [-2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    },
    # ... more constraints
}
```

## Legacy Scripts

Individual constraint scripts are kept in subdirectories for reference:
- `sphere/Poly.py`
- `spiral/{Analytical,NN,poly}.py`
- `cylinder/cylinder.py`
- `hyperplane/Poly.py`

These can be removed once `main_3d.py` is fully validated.

## Troubleshooting

### "No h=0 crossing" in slice visualizations

This means the slice plane doesn't intersect the surface. Adjust `z_levels` and `y_level`:

- **Sphere** (R=1.0): Use Z ∈ [-1, 1]
- **Spiral** (pitch=0.3): Use Z ∈ [-2π×0.3, 2π×0.3] ≈ [-1.9, 1.9]
- **Cylinder** (H=4.0): Use Z ∈ [-2, 2]
- **Hyperplane**: Adjust based on plane equation

### Models not converging

Check in `configs/config3D.py`:
- Increase `EPOCHS`
- Adjust learning rates (`LR_NN`, `LR_POLY`)
- Modify `MARGIN`, `ALPHA_NEG` for loss tuning

## Next Steps

- [ ] Add more architectures (deeper networks, different activations)
- [ ] Implement comparison script for KNN vs Neural Pull
- [ ] Add quantitative metrics module
- [ ] Generate comprehensive comparison report
