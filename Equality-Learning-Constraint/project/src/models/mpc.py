from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import numpy as np

try:
    import osqp
    from scipy import sparse
    _OSQP_AVAILABLE = True
except Exception:
    _OSQP_AVAILABLE = False


@dataclass
class MPCConfig:
    N: int = 10
    dt: float = 0.02
    Q: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 100.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([0.5, 0.5, 0.5]))  # Slightly reduce control cost
    u_min: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0, -1.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    solver: str = "osqp"
    solver_timeout: float = 0.1
    surface_penalty_weight: float = 1000.0  # Soft penalty for h(x)≠0 violations
    # SQP / Augmented Lagrangian parameters
    max_sqp_iters: int = 5
    al_rho: float = 100.0
    al_rho_mult: float = 5.0
    al_tol: float = 1e-3
    use_augmented_lagrangian: bool = True


class MPCController:
    """Basic MPC controller using discrete integrator dynamics and OSQP.

    This implementation assumes simple dynamics x_{k+1} = x_k + u_k * dt and
    constructs a QP in the control sequence u = [u0 ... u_{N-1}] to minimize
    tracking error to a provided reference trajectory.

    If `osqp` is not installed, the solver falls back to returning zero controls.
    """

    def __init__(self, config: Optional[MPCConfig] = None, dynamics_model: Optional[callable] = None):
        self.config = config or MPCConfig()
        # dynamics_model(x, u, dt) -> x_next
        self.dynamics = dynamics_model

    def _dynamics(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        if self.dynamics is not None:
            return self.dynamics(x, u, dt)
        return x + u * dt

    def solve(self, x0: np.ndarray, x_goal: Optional[np.ndarray] = None,
              waypoints: Optional[list] = None, u_prev: Optional[np.ndarray] = None,
              grad_h_fn: Optional[callable] = None, h_fn: Optional[callable] = None,
              grad_obs_fn: Optional[callable] = None, h_obs_fn: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Solve the MPC QP for current state x0.

        Args:
            x0: current state, shape (dim_x,)
            grad_h_fn: gradient for equality constraint (surface: h=0)
            h_fn: implicit function for equality constraint (surface: h=0)
            grad_obs_fn: gradient for inequality constraint (obstacle: h>=0 to stay outside)
            h_obs_fn: implicit function for inequality constraint (obstacle: h>=0)

        Returns:
            u_seq: (N, dim_u)
            x_seq: (N+1, dim_x)
            info: solver info dict
        """
        cfg = self.config
        N = cfg.N
        dim_x = x0.shape[0]
        dim_u = cfg.u_min.shape[0]

        # Build B matrix such that x_seq_without_x0 = B * u_vec
        def build_B(N, dim_x, dim_u, dt):
            rows = []
            cols = []
            data = []
            for k in range(1, N + 1):
                for j in range(0, k):
                    for i in range(dim_u):
                        row = (k - 1) * dim_x + i
                        col = j * dim_u + i
                        rows.append(row)
                        cols.append(col)
                        data.append(dt)
            B = sparse.coo_matrix((data, (rows, cols)), shape=(N * dim_x, N * dim_u))
            return B.tocsc()

        B = build_B(N, dim_x, dim_u, cfg.dt)

        # Stack x0 for later use
        x0_stack = np.tile(x0.reshape(-1, 1), (N, 1)).reshape(N * dim_x)

        # Block-diagonal Q and R
        Qbar = sparse.block_diag([cfg.Q for _ in range(N)]).tocsc()
        Rbar = sparse.block_diag([cfg.R for _ in range(N)]).tocsc()


        # Trajectory optimization: track reference trajectory or reach x_goal
        # If waypoints provided, track them; otherwise only terminal cost
        # Control cost: sum_k u_k^T R u_k
        # Surface constraints enforced via linearized equality constraints
        
        B_dense = B.toarray()
        
        # Build tracking cost based on waypoints or x_goal
        if waypoints is not None and len(waypoints) > 0:
            # Trajectory tracking: penalize deviation from waypoints at each step
            # Cost: sum_k (x_k - waypoint_k)^T Q (x_k - waypoint_k)
            waypoints_array = np.array(waypoints)
            
            # Ensure waypoints has right length (pad or truncate if needed)
            if len(waypoints_array) < N + 1:
                # Pad with last waypoint
                last_wp = waypoints_array[-1]
                padding = np.tile(last_wp, (N + 1 - len(waypoints_array), 1))
                waypoints_array = np.vstack([waypoints_array, padding])
            elif len(waypoints_array) > N + 1:
                # Truncate to N+1
                waypoints_array = waypoints_array[:N+1]
            
            # Stack reference for all timesteps (skip x0)
            xref_stack = waypoints_array[1:].reshape(-1)  # (N * dim_x,)
            
            # Cost: (x - xref)^T Qbar (x - xref) where x = x0_stack + B*u
            # Expanding: (x0_stack + B*u - xref)^T Qbar (x0_stack + B*u - xref)
            # = u^T B^T Qbar B u + 2*(x0_stack - xref)^T Qbar B u + const
            
            # P = 2 * B^T Qbar B + Rbar
            H = sparse.csc_matrix(B.T.dot(Qbar).dot(B)) + Rbar
            P = 2.0 * H
            P = 0.5 * (P + P.T)
            
            # q = 2 * (x0_stack - xref)^T Qbar B
            residual = x0_stack - xref_stack
            q = 2.0 * (B.T.dot(Qbar.dot(residual)))
            q = q.astype(float)
            
            # Also add small terminal cost to ensure reaching final waypoint
            C = B_dense[(N-1)*dim_x:N*dim_x, :]
            Qf_terminal = cfg.Q * 2.0  # Small terminal weight
            x_goal_final = waypoints_array[-1]
            P += 2.0 * sparse.csc_matrix(C.T.dot(Qf_terminal).dot(C))
            q += 2.0 * (C.T.dot(Qf_terminal.dot(x0 - x_goal_final)))
            P = 0.5 * (P + P.T)
            
        else:
            # Terminal cost only: prioritize reaching x_goal
            start = (N - 1) * dim_x
            end = N * dim_x
            C = B_dense[start:end, :]
            Qf = cfg.Q * 10.0  # High terminal cost

            # H = C^T Qf C + Rbar
            H = sparse.csc_matrix(C.T.dot(Qf).dot(C)) + Rbar
            P = 2.0 * H
            P = 0.5 * (P + P.T)

            # q = 2 * C^T Qf (x0 - x_goal)
            q = 2.0 * (C.T.dot(Qf.dot(x0 - np.asarray(x_goal).reshape(-1))))
            q = q.astype(float)
    
        # Control bounds
        u_min_stack = np.tile(cfg.u_min, N)
        u_max_stack = np.tile(cfg.u_max, N)

        # Setup linear constraints for OSQP. By default we only have simple box
        # bounds on controls (u_min <= u <= u_max) represented as identity.
        # Optionally, the caller may provide `grad_h_fn` to add linearized
        # equality constraints of the form A_eq u = b_eq derived from
        # grad_h(x_lin)^T (x0 + B_k u - x_lin) = 0 for k=1..N.

        # Start with identity for control bounds
        A_box = sparse.eye(N * dim_u, format='csc')
        l_box = u_min_stack.astype(float)
        u_box = u_max_stack.astype(float)

        # Build reference trajectory for linearization
        if waypoints is not None and len(waypoints) > 0:
            # Use waypoints as reference trajectory
            waypoints_array = np.array(waypoints)
            if len(waypoints_array) < N + 1:
                # Pad with last waypoint
                last_wp = waypoints_array[-1]
                padding = np.tile(last_wp, (N + 1 - len(waypoints_array), 1))
                xref = np.vstack([waypoints_array, padding])
            elif len(waypoints_array) > N + 1:
                # Truncate to N+1
                xref = waypoints_array[:N+1]
            else:
                xref = waypoints_array
        else:
            # Linear interpolation between x0 and x_goal
            xref = np.linspace(x0.reshape(1, -1), np.asarray(x_goal).reshape(1, -1), N + 1).reshape(N + 1, dim_x)
    
        # If h_fn is provided together with grad_h_fn, project the linearization
        # points onto the surface h(x)=0 using Newton steps to improve
        # the linear approximation (helps enforce the learned surface).
        # More iterations + tighter tolerance = stronger surface enforcement.
        def _find_surface_point(x_init, h_fn_local, grad_h_fn_local, max_iter=30, tol=1e-8):
            """Find point on surface h(x)=0 by moving along gradient direction.
            
            Uses a hybrid approach:
            1. Binary search along gradient to bracket the zero
            2. Bisection to find h(x)=0 with high accuracy
            
            This is more robust than Newton for surfaces with variable curvature.
            """
            x = x_init.astype(float).copy()
            h_init = float(h_fn_local(x))
            
            # If already very close to surface, return
            if abs(h_init) < tol:
                return x
            
            # Get gradient direction (perpendicular to surface)
            grad = np.asarray(grad_h_fn_local(x)).reshape(-1)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < 1e-12:
                return x  # gradient too weak, can't determine direction
            
            grad_unit = grad / grad_norm
            
            # Determine search direction: move toward h=0
            # If h>0, move in -grad direction; if h<0, move in +grad direction
            search_direction = -np.sign(h_init) * grad_unit
            
            # Phase 1: Find bracket [x_low, x_high] where h changes sign
            # Start with approximate step based on |h|/||grad||
            step_size = abs(h_init) / grad_norm
            x_search = x + search_direction * step_size
            h_search = float(h_fn_local(x_search))
            
            # Expand/contract search to bracket the zero
            max_bracket_iter = 10
            for _ in range(max_bracket_iter):
                if h_init * h_search < 0:
                    # Found bracket! h changes sign between x and x_search
                    break
                # Adjust step size
                if abs(h_search) < abs(h_init):
                    # Moving in right direction, increase step
                    step_size *= 2.0
                else:
                    # Overshot or wrong direction, reduce step
                    step_size *= 0.5
                x_search = x + search_direction * step_size
                h_search = float(h_fn_local(x_search))
            
            # Phase 2: Bisection on the bracketed interval
            if h_init * h_search < 0:
                # We have a bracket
                x_low = x.copy()
                x_high = x_search.copy()
                h_low = h_init
                h_high = h_search
                
                for _ in range(max_iter):
                    x_mid = 0.5 * (x_low + x_high)
                    h_mid = float(h_fn_local(x_mid))
                    
                    if abs(h_mid) < tol:
                        return x_mid
                    
                    # Update bracket
                    if h_low * h_mid < 0:
                        x_high = x_mid
                        h_high = h_mid
                    else:
                        x_low = x_mid
                        h_low = h_mid
                
                return x_mid
            else:
                # Couldn't bracket, fall back to Newton step as last resort
                step = h_init / (grad_norm ** 2)
                return x - step * grad

        # SQP / Augmented Lagrangian setup
        max_sqp_iters = cfg.max_sqp_iters
        rho = cfg.al_rho
        rho_mult = cfg.al_rho_mult
        al_tol = cfg.al_tol
        use_al = cfg.use_augmented_lagrangian

        # Initial linearization points (N+1, dim_x)
        x_lin = xref.copy()
        # Fix final point to be exactly x_goal (don't let projection move it)
        x_lin[N] = np.asarray(x_goal).reshape(-1).copy()
        lambda_eq = None

        # Precompute base P and q (constant across iterations)
        P_base = sparse.csc_matrix(P)
        q_base = q.copy()

        last_resid = np.inf
        u_vec = np.zeros(N * dim_u, dtype=float)
        x_seq_candidate = None

        # Iterative SQP loop
        for it in range(max_sqp_iters):
            # Build linearized equality A_eq * u = b_eq using current x_lin
            eq_rows = []
            eq_cols = []
            eq_data = []
            be_list = []
            eq_idx = 0

            B_dense = B.toarray()  # small N expected; convert to dense for convenience

            # Build equality constraints (surface: h=0)
            for k in range(1, N + 1):
                xk_lin = x_lin[k]
                if (h_fn is not None) and (grad_h_fn is not None):
                    try:
                        # Project linearization point onto surface for better approximation
                        xk_lin = _find_surface_point(xk_lin, h_fn, grad_h_fn, max_iter=30, tol=1e-8)
                    except Exception:
                        pass
                    grad = np.asarray(grad_h_fn(xk_lin)).reshape(-1)
                    start_r = (k - 1) * dim_x
                    end_r = k * dim_x
                    B_block = B_dense[start_r:end_r, :]
                    Ae_row = grad.dot(B_block)  # shape (N*dim_u,)
                    nz = np.nonzero(np.abs(Ae_row) > 1e-12)[0]
                    for col in nz:
                        eq_rows.append(eq_idx)
                        eq_cols.append(col)
                        eq_data.append(float(Ae_row[col]))
                    be = - float(grad.dot(x0 - xk_lin))
                    be_list.append(be)
                    eq_idx += 1
            
            # Build soft penalty for obstacle avoidance (instead of hard constraints)
            # Parameters for obstacle penalty
            obstacle_margin = 0.2      # Distance threshold to start penalizing
            w_obs = 500.0              # Weight for obstacle penalty
            
            P_obs = sparse.csc_matrix((N * dim_u, N * dim_u))
            q_obs = np.zeros(N * dim_u)
            
            if (h_obs_fn is not None) and (grad_obs_fn is not None):
                for k in range(1, N + 1):
                    xk_lin = x_lin[k]
                    d_val = float(h_obs_fn(xk_lin))  # Distance to obstacle (h >= 0 outside)
                    
                    # Only penalize if within margin
                    if d_val < obstacle_margin:
                        # Slack variable: s = margin - d (how much we violate the margin)
                        s = obstacle_margin - d_val
                        
                        grad_d = np.asarray(grad_obs_fn(xk_lin)).reshape(-1)  # ∇h wrt x
                        grad_norm = np.linalg.norm(grad_d)
                        
                        # Skip if gradient too small
                        if grad_norm < 1e-8:
                            continue
                        
                        start_r = (k - 1) * dim_x
                        end_r = k * dim_x
                        B_block = B_dense[start_r:end_r, :]
                        gradB = grad_d.dot(B_block)  # shape (N*dim_u,)
                        
                        # Quadratic penalty on slack: w * s^2
                        # Linearize: s ≈ margin - d_val - grad_d^T(x_k - x_lin)
                        #            = margin - d_val - grad_d^T(x0 + B*u - x_lin)
                        #            = const - gradB^T * u
                        const = obstacle_margin - d_val - grad_d.dot(x0 - xk_lin)
                        
                        # Add to cost: w * (const - gradB^T * u)^2
                        # Expanding: w * (const^2 - 2*const*gradB^T*u + u^T*gradB*gradB^T*u)
                        # P contribution: 2*w * gradB*gradB^T
                        # q contribution: -2*w*const * gradB
                        P_obs = P_obs + sparse.csc_matrix(2.0 * w_obs * np.outer(gradB, gradB))
                        q_obs = q_obs - 2.0 * w_obs * const * gradB
            
            # Add control rate penalty (smooth control changes)
            w_rate = 1.0  # Weight for control rate penalty
            rows, cols, vals = [], [], []
            for k in range(N - 1):
                for i in range(dim_u):
                    rows += [k * dim_u + i, k * dim_u + i]
                    cols += [k * dim_u + i, (k + 1) * dim_u + i]
                    vals += [1.0, -1.0]
            
            if len(rows) > 0:
                D = sparse.coo_matrix((vals, (rows, cols)), shape=((N - 1) * dim_u, N * dim_u)).tocsc()
                P_rate = (D.T).dot(D) * w_rate
            else:
                P_rate = sparse.csc_matrix((N * dim_u, N * dim_u))

            # Assemble equality constraints
            if len(be_list) > 0:
                A_eq = sparse.coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(len(be_list), N * dim_u)).tocsc()
                b_eq = np.asarray(be_list, dtype=float)
            else:
                A_eq = None
                b_eq = None

            # Configure QP based on Augmented Lagrangian or hard constraints
            if use_al and (A_eq is not None):
                # Augmented Lagrangian: enforce EQUALITY constraints as quadratic penalty + multipliers
                if lambda_eq is None:
                    lambda_eq = np.zeros(A_eq.shape[0], dtype=float)

                AtA = (A_eq.T).dot(A_eq)  # sparse
                P_aug = P_base + rho * AtA + P_obs + P_rate
                q_aug = q_base + (A_eq.T).dot(lambda_eq) - rho * (A_eq.T).dot(b_eq) + q_obs
                
                # Only box constraints (obstacles handled via soft penalty)
                A_qp = A_box
                l_qp = l_box
                u_qp = u_box
            else:
                # Hard constraints for equality only (obstacles handled via soft penalty)
                if A_eq is not None:
                    A_qp = sparse.vstack([A_eq, A_box]).tocsc()
                    l_qp = np.concatenate([b_eq, l_box]).astype(float)
                    u_qp = np.concatenate([b_eq, u_box]).astype(float)
                else:
                    A_qp = A_box
                    l_qp = l_box
                    u_qp = u_box
                
                P_aug = P_base + P_obs + P_rate
                q_aug = q_base + q_obs

            # Ensure symmetry
            P_qp = sparse.csc_matrix(0.5 * (P_aug + P_aug.T))

            # Setup and solve QP
            prob = osqp.OSQP()
            prob.setup(P=P_qp, q=q_aug.astype(float), A=A_qp, l=l_qp.astype(float), u=u_qp.astype(float),
                       verbose=False, warm_start=True, eps_abs=1e-3, eps_rel=1e-3, max_iter=10000)
            res = prob.solve()

            # Extract solution
            if hasattr(res, "x") and res.x is not None:
                u_vec = res.x
            else:
                u_vec = np.zeros(N * dim_u, dtype=float)

            # Compute constraint residual and update multipliers if using AL
            if use_al and (A_eq is not None):
                resid = A_eq.dot(u_vec) - b_eq
                max_resid = float(np.max(np.abs(resid)))
                # Update Lagrange multipliers
                lambda_eq = lambda_eq + rho * resid

                # Adapt rho if residual not improving sufficiently
                if max_resid > 0.1 * last_resid and max_resid > al_tol:
                    rho *= rho_mult

                last_resid = max_resid
            else:
                # Compute residual for diagnostics even with hard constraints
                if A_eq is not None:
                    resid = A_eq.dot(u_vec) - b_eq
                    max_resid = float(np.max(np.abs(resid)))
                    last_resid = max_resid
                else:
                    max_resid = 0.0

            # Simulate predicted states and update x_lin for next iteration
            u_seq_candidate = u_vec.reshape(N, dim_u)
            x_seq_candidate = np.zeros((N + 1, dim_x), dtype=float)
            x_seq_candidate[0] = x0.copy()
            for k in range(N):
                x_seq_candidate[k + 1] = self._dynamics(x_seq_candidate[k], u_seq_candidate[k], cfg.dt)

            # Project predicted states onto surface for next linearization
            # BUT keep final point fixed to x_goal to ensure exact arrival
            if (h_fn is not None) and (grad_h_fn is not None):
                for k in range(1, N):  # Only project intermediate points (not final point N)
                    try:
                        x_seq_candidate[k] = _find_surface_point(x_seq_candidate[k], h_fn, grad_h_fn, max_iter=30, tol=1e-8)
                    except Exception:
                        pass

            x_lin = x_seq_candidate.copy()
            # Re-fix final point to x_goal after projection
            x_lin[N] = np.asarray(x_goal).reshape(-1).copy()

            # Break if constraint satisfaction achieved
            if max_resid < al_tol:
                break

        # Final solution after SQP loop
        u_seq = u_vec.reshape(N, dim_u)
        x_seq = x_seq_candidate.copy() if x_seq_candidate is not None else np.zeros((N + 1, dim_x))

        info = {
            "status": getattr(res.info, 'status', None),
            "solve_time": getattr(res.info, 'run_time', None),
            "obj_val": getattr(res.info, 'obj_val', None),
            "max_constraint_violation": float(last_resid),
            "sqp_iters": it + 1
        }
        return u_seq, x_seq, info


__all__ = ["MPCConfig", "MPCController"]
