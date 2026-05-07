import sys
import os
import torch

import inspect

HERE = os.path.abspath(os.path.dirname(__file__))                  
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))  
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')                       
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 

print("[DBG] PROJECT_ROOT:", PROJECT_ROOT)
print("[DBG] SRC_DIR exists?:", os.path.isdir(SRC_DIR))
print("[DBG] sys.path[0]:", sys.path[0])

try:
    import src
    print("[DBG] 'src' package loaded from:", os.path.dirname(inspect.getfile(src)))
except Exception as e:
    print("[DBG] Cannot import 'src':", e)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    print("[DBG] Added SRC_DIR to sys.path (importarà com 'from models...' sense 'src.')")

from src.models.base import MLPImplicit, PolyImplicit 
from src.training.train_loop import train_one
from src.data.datasets import make_constraint
from src.data.datasets import sample_negative_from_normals
from src.utils.viz import *
from src.utils.viz2D import *
from src.utils.metrics import grid_zero_contour_points, geometry_chamfer_score
from configs import config

from src.utils.seed import set_seed

set_seed(config.SEED)

archs = [
    ("NN(4x4)",     lambda: MLPImplicit(hidden=4,  layers=2), config.EPOCHS,   config.LR_NN),
    ("NN(16x16)",   lambda: MLPImplicit(hidden=16, layers=2), config.EPOCHS,   config.LR_NN),
    ("Poly(deg=2)", lambda: PolyImplicit(degree=2),           config.EPOCHS, config.LR_POLY),
    ("Poly(deg=3)", lambda: PolyImplicit(degree=3),           config.EPOCHS, config.LR_POLY),
    ("Poly(deg=4)", lambda: PolyImplicit(degree=4),           config.EPOCHS, config.LR_POLY),
]
loss_variants = [
    (1, "POS"),
    (2, "POS+NEG(sp)"),
    (3, "POS+L2"),
    (4, "POS+GRAD"),
]

h_np, grad_np, gt_samples, gt_curve = make_constraint(config.CONSTRAINT_NAME)

pos_np = gt_samples(config.NPOS)
neg_np = sample_negative_from_normals(pos_np, grad_np, h_np, std=0.15)


results = {}
geometric_scores = {}

for arch_name, make_model, epochs, lr in archs:
    for vid, vname in loss_variants:
        model = make_model()
        pos = torch.tensor(pos_np, dtype=torch.float32)
        neg = torch.tensor(neg_np, dtype=torch.float32)
        
        if vid == 3:
            model, curve = train_one(model, pos=pos, neg=neg, variant=vid, epochs=config.EPOCHS, lr=lr, param_l2=config.LAMBDA_L2)
        elif vid == 4:
            model, curve = train_one(model, pos=pos, neg=neg, variant=vid, epochs=config.EPOCHS, lr=lr, grad_norm_penalty=config.LAMBDA_GRAD)
        else:
            model, curve = train_one(model, pos=pos, neg=neg, variant=vid, epochs=config.EPOCHS, lr=lr)
        
        results[(arch_name, vname)] = (model, curve)
        
        gt_dense = gt_curve(M=1000)
        gs, cd = geometry_chamfer_score(
            model, gt_dense, 
            sigma=config.SIGMA_GEOM
        )
        geometric_scores[(arch_name, vname)] = (gs, cd)
        print(f"{arch_name} + {vname}: GS={gs:.3f}, CD={cd:.3f}")

outputs_root = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(outputs_root, exist_ok=True)
script_name = os.path.splitext(os.path.basename(__file__))[0]
fig_dir = os.path.join(outputs_root, f"{script_name}_figures")
os.makedirs(fig_dir, exist_ok=True)

plot_samples(pos_np, neg_np, gt_curve, grid_lim=config.GRID_LIM, save_path=os.path.join(fig_dir, "figC_samples.png"), constraint_name=config.CONSTRAINT_NAME)
plot_levelset_grid(results, archs, loss_variants, gt_curve, grid_lim=config.GRID_LIM, grid_n=config.GRID_N, save_path=os.path.join(fig_dir, "figA_levelsets_4x4.png"), constraint_name=config.CONSTRAINT_NAME)
plot_curve_grid(results, archs, loss_variants, save_path=os.path.join(fig_dir, "figB_curves_4x4.png"))

plot_heatmap(
    archs=archs,
    loss_variants=loss_variants,
    name=config.CONSTRAINT_NAME,
    results=results,
    pos_tensor=torch.tensor(pos_np, dtype=torch.float32),
    config=config,
    grid_zero_contour_points_fn=grid_zero_contour_points,
    save_path=os.path.join(fig_dir, "figD_heatmap.png")
)

print("\n=== RESUM GEOMETRIC SCORES ===")
for (arch_name, vname), (gs, cd) in geometric_scores.items():
    print(f"{arch_name:12} + {vname:12}: GS={gs:.3f}, CD={cd:.3f}")
