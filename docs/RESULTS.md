# Selected Results

This page documents the curated figures included in `assets/`. The full generated experiment outputs are intentionally not tracked in Git.

## Surface Learning

The project learns implicit neural constraints whose zero-level set approximates the demonstrated geometry.

| Figure | Description |
| --- | --- |
| ![Sphere reconstruction](../assets/surface_reconstruction.png) | Sphere reconstruction from trajectory-style samples using an implicit neural model. |
| ![Ellipsoid reconstruction](../assets/ellipsoid_reconstruction.png) | Ellipsoid reconstruction example from the 3D PCA negative-sampling comparison. |
| ![Cylinder reconstruction](../assets/cylinder_reconstruction.png) | Cylinder reconstruction example from the 3D PCA negative-sampling comparison. |

## 2D Experiments

The 2D scripts are useful for quickly inspecting learned level sets, loss behavior, and projection effects.

| Figure | Description |
| --- | --- |
| ![2D level sets](../assets/2d_level_sets.png) | Learned level sets for the 2D comparison experiment. |
| ![2D training curves](../assets/2d_training_curves.png) | Training curves from the 2D loss comparison. |
| ![Projected trajectories](../assets/projected_trajectories.png) | Projection-based correction of off-manifold 2D points. |

## PCA-Based Negative Sampling

Negative samples are generated around demonstrated trajectories using local normal estimates. This gives the implicit model off-surface supervision while keeping the source demonstrations on the target geometry.

| Figure | Description |
| --- | --- |
| ![Sphere training samples](../assets/sphere_training_samples.png) | Positive and generated samples used for 3D constraint learning. |

## Motion Generation

After learning a constraint, the project evaluates trajectory correction and simple constrained motion generation.

| Figure | Description |
| --- | --- |
| ![Sphere planning](../assets/constraint_motion_planning.png) | Constraint-aware planning comparison on a sphere. |
| ![Ellipsoid planning](../assets/ellipsoid_motion_planning.png) | Constraint-aware planning comparison on an ellipsoid. |
| ![Cylinder planning](../assets/cylinder_motion_planning.png) | Constraint-aware planning comparison on a cylinder. |
| ![Trajectory tracking](../assets/trajectory_tracking.png) | Tracking example comparing raw and constrained motion behavior. |

## What Is Not Included

The public repository excludes:

- full `outputs/` and `controller_outputs/` trees;
- trained checkpoints and serialized models;
- generated CSV summaries;
- GIFs and videos;
- PDFs and internal reports;
- manuscript or submission material.

Those artifacts can be archived separately if a full reproducibility bundle is needed.
