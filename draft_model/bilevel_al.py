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
    """Compute Hv where H = ∇²_θθ(loss)."""
    grad_theta = torch.autograd.grad(loss, theta, create_graph=True, retain_graph=True)[0]
    return torch.autograd.grad((grad_theta * v).sum(), theta, retain_graph=True)[0]


def _cg_solve(Hvp_fn, b, niter=10, tol=1e-6):
    """Solve Hh = b via conjugate gradient."""
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
):
    """
    Full bilevel AL solver for one client round.
    Returns (θ*, optimized_Ds_features, optimized_U_features).
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

    lam = 0.0
    B_X_plus, B_A_plus = B.B_plus

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
            loss_ds = torch.log(1 + torch.exp(-margin_ds.clamp(min=-50))).mean()
            reg_in = (lambda_theta_in / (2 * (d_plus_1 ** 2))) * ((theta - zeta_t) ** 2).sum()
            Lin_inner = loss_ds + reg_in

            if n_u > 0 and Ds.Delta_s > 0:
                logits_u = (Xa_u @ theta).squeeze(-1)
                loss_u = torch.log(1 + torch.exp(-logits_u.clamp(min=-50))).mean()
                Lin_inner = Lin_inner + lambda_U * loss_u

            Lin_inner.backward()
            inner_opt.step()

        theta_star = theta.detach().clone().requires_grad_(True)

        # Outer: evaluate fairness and loss on real minibatch B
        g_val = g_EO(theta_star, B_X_plus, B_A_plus)
        Lout_val = L_out(theta_star, B.X, B.A, B.Y,
                         torch.zeros_like(zeta_t), lambda_theta_out, d_plus_1)

        # Compute ∇_θ Φ
        theta_star.retain_grad()
        Lout_val.backward(retain_graph=True)
        v = theta_star.grad.clone()
        theta_star.grad = None
        g_val.backward(retain_graph=True)
        v = v + lam * theta_star.grad + rho * g_val.detach() * theta_star.grad
        theta_star.grad = None

        # Solve h = H⁻¹v via CG
        def Hvp(h):
            Xa_ds = torch.cat([X_ds, torch.tensor(Ds.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
            Xa_u = torch.cat([X_u, torch.tensor(U.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
            loss_ds = torch.log(1 + torch.exp(-(2*y_ds-1)*(Xa_ds@theta_star).squeeze(-1).clamp(min=-50))).mean()
            reg = (lambda_theta_in/(2*(d_plus_1**2)))*((theta_star-zeta_t)**2).sum()
            L = loss_ds + reg
            if n_u > 0:
                L = L + lambda_U * torch.log(1+torch.exp(-(Xa_u@theta_star).squeeze(-1).clamp(min=-50))).mean()
            return _hessian_vector_product(L, theta_star, h)

        h = _cg_solve(Hvp, v.float(), niter=5, tol=1e-4)

        # Implicit gradient: update synthetic and Universum features
        Xa_ds = torch.cat([X_ds, torch.tensor(Ds.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
        Xa_u = torch.cat([X_u, torch.tensor(U.A.reshape(-1, 1), dtype=torch.float32, device=device)], dim=1)
        loss_ds = torch.log(1 + torch.exp(-(2*y_ds-1)*(Xa_ds@theta_star).squeeze(-1).clamp(min=-50))).mean()
        reg = (lambda_theta_in/(2*(d_plus_1**2)))*((theta_star-zeta_t)**2).sum()
        Lin_for_x = loss_ds + reg
        if n_u > 0:
            Lin_for_x = Lin_for_x + lambda_U * torch.log(1+torch.exp(-(Xa_u@theta_star).squeeze(-1).clamp(min=-50))).mean()

        grad_theta_lin = torch.autograd.grad(Lin_for_x, theta_star, create_graph=True, retain_graph=True)[0]
        w = (grad_theta_lin * h).sum()
        grad_X_ds = torch.autograd.grad(w, X_ds, allow_unused=True, retain_graph=True)[0]
        grad_X_u = torch.autograd.grad(w, X_u, allow_unused=True)[0]

        if grad_X_ds is not None:
            X_ds.data.sub_(eta_x * grad_X_ds)
            X_ds.data.clamp_(-R, R)
        if grad_X_u is not None:
            X_u.data.sub_(eta_x * grad_X_u)
            X_u.data.clamp_(-R, R)

        # Update Lagrange multiplier
        lam = lam + rho * g_val.detach().item()
        if abs(g_val.item()) <= epsilon_EO:
            break

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
