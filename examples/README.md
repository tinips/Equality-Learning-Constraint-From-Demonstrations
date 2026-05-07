# Examples

This folder is reserved for small, public examples only. It intentionally does not contain datasets, checkpoints, generated plots, videos, logs, or private research materials.

Run the main scripts from the `project/` directory:

```bash
cd project
python experiments/3D/main_3d.py --constraint sphere --dataset trajectory --traj-num 35 --traj-points 25 --regularizer eikonal --no-show
python experiments/3D/cont.py --constraint sphere --mpc --no-show
```

Configuration defaults live in `project/configs/`.
