"""
Algorithm 1: Client-side Augmented Lagrangian bilevel solver.

Inner level: minimize Lin(θ) over θ using synthetic + Universum data.
Outer level: minimize Φ over synthetic/Universum features with EO
constraint via implicit differentiation (H⁻¹v solved by CG).

Run via: python -m draft_model.run_draft (not directly)
"""
import numpy as np
import torch
from .notation import OriginalMinibatch, SyntheticMinibatch, UniversumSet
from .losses import L_out, g_EO


def _hessian_vector_product(loss, theta, v):
    """Compute Hv where H = ∇²_θθ(loss), via the Pearlmutter trick (two backward passes
    instead of ever materializing H): first get grad_theta = ∇_θ loss with create_graph=True,
    then differentiate the scalar (grad_theta · v) w.r.t. theta again to get Hv.
    """
    grad_theta = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True)[0]
    return torch.autograd.grad((grad_theta * v).sum(), theta, retain_graph=True)[0]


def _cg_solve(Hvp_fn, b, niter=10, tol=1e-6):
    """Solve Hh = b via conjugate gradient, using Hvp_fn(x) -> Hx instead of forming H
    explicitly (H is the inner-loss Hessian at theta*; see _hessian_vector_product)."""
    h = torch.zeros_like(b)
    r = b - Hvp_fn(h)
    p = r.clone()
    rs_old = (r * r).sum().item()
    for _ in range(niter):
        Hp = Hvp_fn(p)
        alpha = rs_old / ((p * Hp).sum().item() + 1e-10)
        h = h + alpha * p
        r = r - alpha * Hp
        rs_new = (r * r).sum().item()
        if rs_new < tol:
            break
        p = r + (rs_new / (rs_old + 1e-10)) * p
        rs_old = rs_new
    return h


def exponential_moving_average(current: float, prev: float, beta: float = 0.15) -> float:
    """EMA smoothing for EO estimates."""
    return float((1.0 - beta) * prev + beta * current)


def compute_ema_estimate(g_current: float, g_ema_prev: float, beta: float = 0.15) -> float:
    """Alias used by caller to keep the intent explicit."""
    return exponential_moving_average(g_current, g_ema_prev, beta=beta)


def importance_weights(A: np.ndarray, Y: np.ndarray, clip_value: float = 10.0) -> np.ndarray:
    """
    Importance weights on Y=1 slice: w_i = 1 / P(S=s | Y=1), clipped for stability.
    Returns a vector of shape (n,).
    """
    A = np.asarray(A).astype(int)
    Y = np.asarray(Y)
    w = np.ones(len(Y), dtype=np.float64)
    mask_pos = Y == 1
    if np.any(mask_pos):
        p0 = np.mean(A[mask_pos] == 0)
        p1 = np.mean(A[mask_pos] == 1)
        p = np.array([p0, p1], dtype=np.float64)
        p = np.clip(p, 1e-6, 1.0)
        for i in range(len(Y)):
            if Y[i] == 1:
                w[i] = 1.0 / p[A[i]]
    return np.clip(w, 1.0, clip_value)


def rolling_positive_buffer(B_X: np.ndarray, B_A: np.ndarray, B_Y: np.ndarray, buffer_size: int = 50) -> dict:
    """
    Build/refresh a simple rolling buffer of positive examples per group from B.
    Returns {0: np.ndarray, 1: np.ndarray} of feature rows.
    """
    out = {}
    for s in (0, 1):
        rows = B_X[(B_Y == 1) & (B_A == s)]
        if rows.shape[0] > buffer_size:
            rows = rows[-buffer_size:]
        out[s] = rows
    return out


def client_round_al(
    B: OriginalMinibatch,
    Ds: SyntheticMinibatch,
    U: UniversumSet,
    zeta: np.ndarray,
    lambda_theta_in: float,
    lambda_theta_out: float,
    lambda_U: float,
    rho: float,
    epsilon_EO: float,
    K_inner: int,
    J_outer: int,
    eta_theta: float = 0.01,
    eta_x: float = 0.01,
    R: float = 10.0,
    device=None,
    seed: int = 42,
    debug_invariants: bool = False,
    use_tpr_gap: bool = True,
    tpr_alpha: float = 10.0,
    tpr_tau: float = 0.0,
    ema_beta: float = 0.15,
    use_importance_weighting: bool = True,
    use_rolling_buffer: bool = True,
    w_clip: float = 10.0,
    stop_criterion: str = "eo_gap",
    outer_tol_xhat: float = 1e-6,
):
    """
    Full bilevel AL solver for one client round (Algorithm 1).

    Outer loop (J_outer iters): inner-optimize theta on Ds ∪ U (K_inner Adam steps), then
    take an implicit-differentiation step on the synthetic/Universum features to reduce the
    EO-gap-augmented outer objective Phi = L_out(theta*) + lam*g + (rho/2)*g^2, where theta*
    is treated as an implicit function of those features (via the inner optimality condition).
    The Lagrange multiplier `lam` and its EMA-smoothed fairness signal `g_ema` are updated
    each outer iteration; the loop stops early once the EO gap (or the outer feature-gradient
    norm, depending on `stop_criterion`) is within tolerance.

    Args:
        B: real client minibatch (fairness is evaluated against this).
        Ds: synthetic minibatch with fixed (a, y) labels and learnable features.
        U: Universum pseudo-positive set (also feature-learnable).
        zeta: global model broadcast from the server (inner-loop regularization anchor).
        rho, epsilon_EO: fairness penalty strength / EO-gap tolerance for early stopping.
        K_inner, J_outer: inner Adam steps per outer iteration / max outer iterations.

    Returns:
        (theta_star, updated Ds features, updated U features) as numpy arrays — the payload
        the client sends to the server (never the raw data B).
    """
    if device is None:
        device = torch.device("cpu")
    d = B.X.shape[1]
    d_plus_1 = d + 1

    n_u = U.size()
    X_ds = torch.tensor(Ds.X.copy(), dtype=torch.float32, device=device, requires_grad=True)
    X_u = torch.tensor(U.X.copy(), dtype=torch.float32, device=device, requires_grad=True)
    zeta_t = torch.tensor(zeta, dtype=torch.float32, device=device)
    theta = zeta_t.clone().detach().requires_grad_(True)

    torch.manual_seed(seed)
    lam = 0.0
    g_ema = 0.0
    grad_inf_x0 = None
    pos_buffer = rolling_positive_buffer(B.X, B.A, B.Y, buffer_size=50) if use_rolling_buffer else {0: np.zeros((0, d)), 1: np.zeros((0, d))}
    ds_weights_np = importance_weights(Ds.A, Ds.Y, clip_value=w_clip) if use_importance_weighting else np.ones_like(Ds.Y, dtype=np.float64)
    ds_weights_t = torch.tensor(ds_weights_np, dtype=torch.float32, device=device)

    for j in range(J_outer):
        # Inner: train θ on Dˢ ∪ U (features detached)
        inner_opt = torch.optim.Adam([theta], lr=eta_theta)
        for _ in range(K_inner):
            inner_opt.zero_grad()
            A_ds = torch.tensor(Ds.A.reshape(-1, 1), dtype=torch.float32, device=device)
            A_u = torch.tensor(U.A.reshape(-1, 1), dtype=torch.float32, device=device)
            Xa_ds = torch.cat([X_ds.detach(), A_ds], dim=1)
            Xa_u = torch.cat([X_u.detach(), A_u], dim=1)
            y_ds = torch.tensor(Ds.Y, dtype=torch.float32, device=device)

            logits_ds = (Xa_ds @ theta)
            margin_ds = (2 * y_ds - 1) * logits_ds.squeeze(-1)
            loss_per = torch.log(1 + torch.exp(-margin_ds.clamp(min=-50)))
            loss_ds = (loss_per * ds_weights_t).sum() / (ds_weights_t.sum() + 1e-12)
            reg_in = (lambda_theta_in / (2 * (d_plus_1 ** 2))) * ((theta - zeta_t) ** 2).sum()
            Lin_inner = loss_ds + reg_in

            if n_u > 0 and Ds.Delta_s > 0:
                logits_u = (Xa_u @ theta).squeeze(-1)
                loss_u = torch.log(1 + torch.exp(-logits_u.clamp(min=-50))).mean()
                Lin_inner = Lin_inner + lambda_U * loss_u

            Lin_inner.backward()
            inner_opt.step()

        theta_star = theta.detach().clone().requires_grad_(True)

        # Outer: evaluate fairness on real minibatch B (plus optional buffer fill for missing positives).
        B_X_fair, B_A_fair, B_Y_fair = B.X, B.A, B.Y
        if use_rolling_buffer:
            need_append = []
            for s in (0, 1):
                has_pos = np.any((B_A_fair == s) & (B_Y_fair == 1))
                if not has_pos and pos_buffer[s].shape[0] > 0:
                    X_fill = pos_buffer[s]
                    A_fill = np.full(X_fill.shape[0], s, dtype=np.float64)
                    Y_fill = np.ones(X_fill.shape[0], dtype=np.float64)
                    need_append.append((X_fill, A_fill, Y_fill))
            if need_append:
                B_X_fair = np.vstack([B_X_fair] + [x[0] for x in need_append])
                B_A_fair = np.concatenate([B_A_fair] + [x[1] for x in need_append])
                B_Y_fair = np.concatenate([B_Y_fair] + [x[2] for x in need_append])

        g_val = g_EO(
            theta_star, B_X_fair, B_A_fair, B_Y_fair,
            tau=tpr_tau if use_tpr_gap else 0.0,
            alpha=tpr_alpha if use_tpr_gap else 10.0,
        )
        # Paper (Sec 3.1): L_out(θ) = L_{k,t}(θ; B_{k,t}, 0) — outer ridge is ||θ||², not ||θ-ζ||²
        Lout_val = L_out(theta_star, B.X, B.A, B.Y,
                         torch.zeros_like(zeta_t), lambda_theta_out, d_plus_1)

        # Compute ∇_θ Φ where Φ = L_out(θ) + λ·g(θ) + (ρ/2)·g(θ)²  (augmented Lagrangian).
        # By the chain rule, ∇_θΦ = ∇_θL_out + λ·∇_θg + ρ·g·∇_θg — accumulate each term's
        # gradient into `v` via separate backward passes (clearing .grad between them).
        theta_star.retain_grad()
        Lout_val.backward(retain_graph=True)
        v = theta_star.grad.clone()
        theta_star.grad = None
        g_val.backward(retain_graph=True)
        v = v + lam * theta_star.grad + rho * g_val.detach() * theta_star.grad
        theta_star.grad = None

        # Solve h = H⁻¹v via CG (H = ∇²_θθ Lin at theta*): h is the implicit-differentiation
        # direction used below to propagate ∇_θΦ back onto the Ds/U features.
        def Hvp(h):
            Xa_ds = torch.cat([X_ds, torch.tensor(Ds.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
            Xa_u = torch.cat([X_u, torch.tensor(U.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
            loss_ds = torch.log(1 + torch.exp(-(2*y_ds-1)*(Xa_ds@theta_star).squeeze(-1).clamp(min=-50))).mean()
            reg = (lambda_theta_in/(2*(d_plus_1**2)))*((theta_star-zeta_t)**2).sum()
            L = loss_ds + reg
            if n_u > 0 and Ds.Delta_s > 0:
                L = L + lambda_U * torch.log(1+torch.exp(-(Xa_u@theta_star).squeeze(-1).clamp(min=-50))).mean()
            return _hessian_vector_product(L, theta_star, h)

        h = _cg_solve(Hvp, v.float(), niter=30, tol=1e-4)

        # Implicit gradient: update synthetic and Universum features
        Xa_ds = torch.cat([X_ds, torch.tensor(Ds.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
        Xa_u = torch.cat([X_u, torch.tensor(U.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
        loss_ds = torch.log(1 + torch.exp(-(2*y_ds-1)*(Xa_ds@theta_star).squeeze(-1).clamp(min=-50))).mean()
        reg = (lambda_theta_in/(2*(d_plus_1**2)))*((theta_star-zeta_t)**2).sum()
        Lin_for_x = loss_ds + reg
        if n_u > 0 and Ds.Delta_s > 0:
            Lin_for_x = Lin_for_x + lambda_U * torch.log(1+torch.exp(-(Xa_u@theta_star).squeeze(-1).clamp(min=-50))).mean()

        # Implicit function theorem: d(theta*)/d(x) = -H^-1 * d/dx(∇_θ Lin), so the hypergradient
        # of Phi w.r.t. the features is -(∇_x ∇_θ Lin) @ h. Rather than forming the
        # (features x theta) Jacobian, use the vector-Jacobian trick: differentiate the scalar
        # w = (∇_θ Lin) . h w.r.t. X_ds/X_u directly (grad_theta_lin kept differentiable via
        # create_graph=True above) to get the same product in one backward pass each.
        grad_theta_lin = torch.autograd.grad(Lin_for_x, theta_star, create_graph=True, retain_graph=True)[0]
        w = (grad_theta_lin * h).sum()
        grad_X_ds = torch.autograd.grad(w, X_ds, allow_unused=True, retain_graph=True)[0]
        grad_X_u = torch.autograd.grad(w, X_u, allow_unused=True)[0]

        grad_inf_x = 0.0
        if grad_X_ds is not None:
            grad_inf_x = max(grad_inf_x, torch.max(torch.abs(grad_X_ds)).item())
        if grad_X_u is not None:
            grad_inf_x = max(grad_inf_x, torch.max(torch.abs(grad_X_u)).item())
        if grad_inf_x0 is None:
            grad_inf_x0 = grad_inf_x
        grad_tol = outer_tol_xhat * max(1.0, float(grad_inf_x0))

        if grad_X_ds is not None:
            X_ds.data.sub_(eta_x * grad_X_ds)
            X_ds.data.clamp_(-R, R)
        if grad_X_u is not None:
            X_u.data.sub_(eta_x * grad_X_u)
            X_u.data.clamp_(-R, R)

        # Update EMA-smoothed fairness signal and multiplier
        g_ema = compute_ema_estimate(g_val.detach().item(), g_ema, beta=ema_beta)
        lam = lam + rho * g_ema
        if stop_criterion == "eo_gap":
            if abs(g_ema) <= epsilon_EO:
                break
        elif stop_criterion == "grad_inf":
            if grad_inf_x <= grad_tol:
                break
        else:
            raise ValueError(f"Unknown stop_criterion '{stop_criterion}'. Use 'eo_gap' or 'grad_inf'.")

    return (
        theta_star.detach().cpu().numpy(),
        X_ds.detach().cpu().numpy(),
        X_u.detach().cpu().numpy(),
    )


def client_round_simplified(
    B: OriginalMinibatch,
    Ds: SyntheticMinibatch,
    U: UniversumSet,
    zeta: np.ndarray,
    lambda_theta_in: float,
    lambda_U: float,
    K_inner: int,
    eta_theta: float = 0.05,
    device=None,
):
    """Simplified path: inner-only θ optimization (no feature updates)."""
    if device is None:
        device = torch.device("cpu")
    d = B.X.shape[1]
    d_plus_1 = d + 1
    theta = torch.tensor(zeta, dtype=torch.float32, device=device, requires_grad=True)
    zeta_t = torch.tensor(zeta, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([theta], lr=eta_theta)

    from .losses import Lin
    for _ in range(K_inner):
        opt.zero_grad()
        L = Lin(theta, Ds.X, Ds.A, Ds.Y, U.X, U.A,
                zeta_t, lambda_theta_in, lambda_U, d_plus_1, Ds.Delta_s)
        L.backward()
        opt.step()

    return theta.detach().cpu().numpy()
