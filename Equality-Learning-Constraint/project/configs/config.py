SEED         = 123
GRID_LIM     = 2.5          # viewport: [-GRID_LIM, GRID_LIM]^2
GRID_N       = 400  # grid resolution for contouring & chamfer
NPOS         = 2000
NNEG         = 2000
EPOCHS       = 1500
BATCH        = 128
LR_NN        = 1e-3
LR_POLY      = 5e-3
MARGIN       = 0.65  # softplus margin target (smaller => smoother)
ALPHA_NEG    = 0.5          # weight on negative loss
LAMBDA_L2    = 1e-4
LAMBDA_GRAD  = 1e-3
# scale-invariant metric hyperparams
TAU_DIST     = 0.03          # tolerated geometric distance on data (NRD)
SIGMA_GEOM   = 0.06          # chamfer->score scale
ALPHA_FINAL  = 0.5           # blend DataSat & Geom scores
CONSTRAINT_NAME = "lemniscate"  
EVAL_SPLIT = 0.2       # fraction of data for eval (rest for training)
