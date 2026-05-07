
import sys
import os
import torch

import inspect

HERE = os.path.abspath(os.path.dirname(__file__))                 
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..','..'))  
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')                        
TRAIN_NOISE_LEVEL = [0.03]  

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


from src.models.base import MLPImplicit, PolyImplicit, HybridImplicit
from src.training.train_loop import train_one_with_split
from src.data.datasets import make_constraint, split_train_test
from src.data.datasets import sample_negative_from_normals
from src.utils.viz import *
from src.utils.viz2D import *
from configs import config


from src.utils.seed import set_seed

set_seed(config.SEED)

archs = [
    ("Poly(deg=4)", lambda: PolyImplicit(degree=4), config.EPOCHS, config.LR_POLY)
]
loss_variants = [
    (2, "Margin Loss"),
    (5, "KNN Loss"),
    (6, "Neural Pull Loss") 
]

print("=== EXPERIMENT: KNN vs Neural Pull (NPL) ===")
print("Models:", [arch[0] for arch in archs])
print("Loss variants:", [f"{v[0]}: {v[1]}" for v in loss_variants])
print("Constraint:", config.CONSTRAINT_NAME)
print("="*50)


h_np, grad_np, gt_samples, gt_curve = make_constraint(config.CONSTRAINT_NAME)



pos_np = gt_samples(config.NPOS)  
neg_np = sample_negative_from_normals(pos_np, grad_np, h_np, std=0.15)
results = {}
results_noise=[]
for noise in TRAIN_NOISE_LEVEL:
    print(f"\n--- Training with noise level: {noise} ---")
        
    for arch_name, make_model, epochs, lr in archs:
        for vid, vname in loss_variants:
            print(f"Training {arch_name} with {vname}...")
            model = make_model()
            pos_train, pos_eval, neg_train, neg_eval = split_train_test(pos_np, neg_np, eval_split=config.EVAL_SPLIT, noise_level=noise)
            if vid == 6:
                model, train_curve,test_curve, projection_info = train_one_with_split(
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
                is_2d=True
            )
            else:   
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
            final_train_loss = train_curve[-1] if train_curve else 0
            final_test_loss = test_curve[-1] if test_curve else 0
            print(f"  Final train loss: {final_train_loss:.6f}, Final test loss: {final_test_loss:.6f}")
            
            with torch.no_grad():
               
                x_test = torch.linspace(-2.5, 2.5, 100)
                y_test = torch.linspace(-2.5, 2.5, 100)
                X, Y = torch.meshgrid(x_test, y_test, indexing='ij')
                grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
                grid_vals = model(grid_points)
                print(f"  Model values on grid: min={grid_vals.min():.4f}, max={grid_vals.max():.4f}")
                print(f"  Values near zero (|f| < 0.1): {(grid_vals.abs() < 0.1).sum().item()}/{len(grid_vals)}")
                
            results[(arch_name, vname)] = (model, train_curve,test_curve)
    
    print(f"\n--- Generating plots for noise level: {noise} ---")
    outputs_root = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outputs_root, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    fig_dir = os.path.join(outputs_root, f"{script_name}_figures_noise_{noise}")
    os.makedirs(fig_dir, exist_ok=True)
    
    print("  Generating curves plot...")
    plot_curve_grid(results, archs, loss_variants, save_path=os.path.join(fig_dir, "figB_curves_4x4.png"))

    print("  Generating heatmap...")
    plot_heatmap(
        archs=archs,
        loss_variants=loss_variants,
        name=config.CONSTRAINT_NAME,
        results=results,
        pos_tensor=pos_eval,
        neg_tensor=neg_eval,
        config=config,
        save_path=os.path.join(fig_dir, "figD_heatmap.png"),
        )
    
    print("  Generating confusion matrix...")
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

    print("  Generating levelset with samples grid...")
    
    plot_levelset_with_samples(
        results=results,
        archs=archs,
        loss_variants=loss_variants,
        gt_curve_fn=gt_curve,
        pos_np=pos_train,
        neg_np=neg_train,
        grid_lim=config.GRID_LIM,
        grid_n=config.GRID_N,
        constraint_name=config.CONSTRAINT_NAME,
        save_path=os.path.join(fig_dir, "figI_levelsets_with_samples_grid.png")
    )
        

    print("  Generating projection visualization...")
    plot_projection_visualization(projection_info, gt_curve, grid_lim=2.5, save_path=os.path.join(fig_dir, "figJ_projection_visualization_adaptative_LF2.png"), n_show=50)
        

     

    
    print(f"  Plots saved to: {fig_dir}")
