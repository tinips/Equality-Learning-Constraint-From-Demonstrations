
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import inspect

HERE = os.path.abspath(os.path.dirname(__file__))                 
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..')) 
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')                        
TRAIN_NOISE_LEVEL = [0, 0.01, 0.02, 0.03, 0.05]  

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
from src.training.train_loop import train_one_with_split
from src.data.datasets import make_constraint, split_train_test
from src.data.datasets import sample_negative_from_normals
from src.utils.viz import *
from src.utils.viz2D import *
from src.utils.metrics import grid_zero_contour_points
from configs import config


from src.utils.seed import set_seed

set_seed(config.SEED)

archs = [
    ("Poly(deg=4)", lambda: PolyImplicit(degree=4),           config.EPOCHS, config.LR_POLY)
]
loss_variants = [
   
    (5, "POS+KNN-NEG")

]


h_np, grad_np, gt_samples, gt_curve = make_constraint(config.CONSTRAINT_NAME)



pos_np = gt_samples(config.NPOS)
neg_np = sample_negative_from_normals(pos_np, grad_np, h_np, std=0.15)

results = {}
results_noise=[]
for noise in TRAIN_NOISE_LEVEL:
        
    for arch_name, make_model, epochs, lr in archs:
        for vid, vname in loss_variants:
            model = make_model()
            pos_train, pos_eval, neg_train, neg_eval = split_train_test(pos_np, neg_np, eval_split=config.EVAL_SPLIT, noise_level=noise)
           
            model, train_curve,test_curve = train_one_with_split(
                model,
                pos_train,
                neg_train,
                pos_eval,
                neg_eval,
                variant=vid,
                epochs=config.EPOCHS,
                lr=lr,
                batch=config.BATCH,
                margin=config.MARGIN,
                alpha=config.ALPHA_NEG,
                
            )
            results[(arch_name, vname)] = (model, train_curve,test_curve)
    outputs_root = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_root, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    noise_root = os.path.join(outputs_root, "noise")
    fig_dir = os.path.join(noise_root, f"{script_name}_figures_noise_{noise}")
    os.makedirs(fig_dir, exist_ok=True)
    plot_curve_grid(results, archs, loss_variants, save_path=os.path.join(fig_dir, "figB_curves_4x4.png"))

    plot_heatmap(
        archs=archs,
        loss_variants=loss_variants,
        name=config.CONSTRAINT_NAME,
        results=results,
        pos_tensor=pos_eval,
        neg_tensor=neg_eval,
        config=config,
        grid_zero_contour_points_fn=grid_zero_contour_points,
        save_path=os.path.join(fig_dir, "figD_heatmap.png"),
        )
    plot_confusion_matrix_grid(
    results,
    archs,
    loss_variants,
    pos_eval,
    neg_eval,
    save_path=os.path.join(fig_dir, "figE_confusion_matrices.png"),
    pos_eps=0.03,
    neg_thr=0.08
)
    plot_levelset_with_samples(
                model=model,
                gt_curve_fn=gt_curve,
                pos_np=pos_train,        
                neg_np=neg_train,        
                grid_lim=config.GRID_LIM,
                grid_n=config.GRID_N,
                title=f"Level Set with ALL Samples\n{arch_name} + {vname}",
                constraint_name=config.CONSTRAINT_NAME,
                save_path=os.path.join(fig_dir, "figF_levelset_with_all_samples.png")
            )