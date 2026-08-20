"""Supplementary numerical code for the associated manuscript. The script constructs the Lieb-Liniger equation of state using product integration of the Bethe kernel, then minimizes a boundary-fitted hydrodynamic action using an Adam/L-BFGS scheme"""

import math
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.interpolate
import scipy.linalg
import torch



# ================================================================
# GLOBAL PARAMS
# ================================================================

c = 0.01  # e.g.: 0.1, 1, 10, 100
R = 1.0
n0 = 1.0

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# build equation of state

NQ_EOS = 900
NTHETA_EOS = 1400
N_TABLE_MAX = 20.0 * n0
REBUILD_EOS = True
EOS_CACHE_DIR = "./eos_cache"

# domain settings
YMAX = 7.0 * R
TAU_FACTOR = 7

# boundary shape settings
N_SHAPE_MODES = 10
ETA_MAX = 1

# action from area of the hole included in action
INCLUDE_HOLE_ACTION = True

# tau policy
ADAM_OPTIMIZE_TAU = True
ADAM_TAU_LR = 2.0e-6

# optimisation params
PRINT_EVERY = 500
LBFGS_RESTARTS_PER_CYCLE = 1
LBFGS_HISTORY_SIZE = 80

# minimisatoin procedure, as (NR, NY, cycles, Adam steps/cycle, L-BFGS steps/restart).
LEVELS = [
    (200, 200, 2, 30000, 10000),
    (200, 200, 3, 10000, 10000),
    (200, 200, 20, 0, 15000),
]

# control parameters
EXPONENT_MIN = -14.0
EXPONENT_MAX = 9.0
DENSITY_OVERFLOW_CAP = 18.0 * n0


# ================================================================
# BUILD BETHE ANSATZ EQUATION OF STATE
# ================================================================

def solve_ll_at_q_product(c, q, n_theta):
    """
    Solve the Lieb-Liniger integral equation at fixed rapidity cutoff q.

    Equation:
        2 pi rho(theta)
        =
        1 + int_{-q}^{q} K(theta-theta') rho(theta') dtheta'

    Kernel:
        K(theta-theta') = 2c/[c^2+(theta-theta')^2]

    Robust discretization:
        use exact panel integrals of K over theta' cells.

    For cell [a_j,b_j],

        int_{a_j}^{b_j} 2c/[c^2+(theta_i-theta')^2] dtheta'
        =
        2[atan((b_j-theta_i)/c)-atan((a_j-theta_i)/c)].
    """

    if q <= 0.0:
        return 0.0, 0.0, None, None

    edges = np.linspace(-q, q, n_theta + 1)
    theta = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    upper = (edges[1:][None, :] - theta[:, None]) / c
    lower = (edges[:-1][None, :] - theta[:, None]) / c

    Kint = 2.0 * (np.arctan(upper) - np.arctan(lower))

    A = 2.0 * np.pi * np.eye(n_theta) - Kint
    rhs = np.ones(n_theta)

    rho = scipy.linalg.solve(
        A,
        rhs,
        assume_a="gen",
        check_finite=False,
    )

    n = np.sum(widths * rho)
    e = np.sum(widths * 0.5 * theta**2 * rho)

    return n, e, theta, rho


def build_ll_eos_product(c, n0, n_table_max, n_q=600, n_theta=900):
    """
    Build e(n) from the Lieb-Liniger Bethe ansatz.

    Small-c scaling:
        q(n) ~ 2 sqrt(c n)

    Tonks scaling:
        q(n) ~ pi n

    The q-grid is clustered near q=0 and around q(n0).
    """

    def solve_n_e(q):
        n, e, _, _ = solve_ll_at_q_product(c, q, n_theta)
        return n, e

    q_weak = 3.0 * math.sqrt(max(c * n_table_max, 1.0e-300))
    q_tonks = 1.5 * math.pi * n_table_max
    qmax = max(1.0e-10, min(q_weak, q_tonks))

    for _ in range(40):
        n_end, _ = solve_n_e(qmax)
        if n_end >= 1.05 * n_table_max:
            break
        qmax *= 1.5
    else:
        print("WARNING: qmax expansion did not reach N_TABLE_MAX.")

    print(f"EOS qmax = {qmax:.10g}")

    s = np.linspace(0.0, 1.0, n_q)

    if c < 0.03:
        power = 4.5
    elif c < 0.1:
        power = 4.0
    elif c < 0.5:
        power = 3.0
    else:
        power = 2.0

    qs_main = qmax * s**power

    q0_est = 2.0 * math.sqrt(max(c * n0, 1.0e-300))
    q_focus = q0_est * np.exp(
        np.linspace(np.log(0.05), np.log(12.0), max(40, n_q // 5))
    )
    q_focus = q_focus[(q_focus > 0.0) & (q_focus < qmax)]

    qs = np.unique(np.r_[qs_main, q_focus])
    qs = qs[qs > 0.0]

    ns = []
    es = []
    bad = 0

    for k, q in enumerate(qs):
        n, e = solve_n_e(q)

        if not np.isfinite(n) or not np.isfinite(e):
            bad += 1
            continue

        ns.append(n)
        es.append(e)

        if k % max(1, len(qs)//10) == 0:
            print(f"  q step {k:5d}/{len(qs)}: q={q:.5g}, n={n:.5g}, e={e:.5g}")

    if bad:
        print(f"WARNING: skipped {bad} nonfinite EOS points.")

    ns = np.asarray(ns)
    es = np.asarray(es)

    ns = np.r_[0.0, ns]
    es = np.r_[0.0, es]

    order = np.argsort(ns)
    ns = ns[order]
    es = es[order]

    
    keep = np.r_[True, np.diff(ns) > 1.0e-10] #remove duplicate
    ns = ns[keep]
    es = es[keep]

    if ns[-1] < n_table_max:
        print("")
        print("WARNING: EOS table does not reach requested N_TABLE_MAX.")
        print(f"Reached n_max = {ns[-1]:.8g}, requested {n_table_max:.8g}.")
        print("Increase qmax expansion or reduce N_TABLE_MAX.")
        print("")

    return ns, es, qs, qmax


def local_poly_eos_derivatives(n, e, n0, window_frac=0.35, deg=5):
    """
    Extract e(n0), mu0=e'(n0), and e''(n0) by local polynomial fit.

    This is more stable than differentiating a noisy global spline.
    """

    n = np.asarray(n)
    e = np.asarray(e)

    width = window_frac * max(n0, 1.0e-12)
    mask = (n > n0 - width) & (n < n0 + width)

    if np.sum(mask) < deg + 3:
        idx = np.argsort(np.abs(n - n0))[:max(deg + 5, 20)]
        mask = np.zeros_like(n, dtype=bool)
        mask[idx] = True

    nn = n[mask]
    ee = e[mask]

    scale = max(np.max(np.abs(nn - n0)), 1.0e-12)
    z = (nn - n0) / scale

    # Nearby points get larger weight.
    w = 1.0 / (1.0 + (np.abs(z) / 0.65) ** 6)

    deg_eff = min(deg, len(nn) - 1)
    coeff = np.polynomial.polynomial.polyfit(z, ee, deg_eff, w=w)

    e0 = coeff[0]
    mu0 = coeff[1] / scale if deg_eff >= 1 else np.nan
    e2 = 2.0 * coeff[2] / scale**2 if deg_eff >= 2 else np.nan

    return e0, mu0, e2, nn, ee, coeff, scale


def load_or_build_eos():
    safe_c = str(c).replace(".", "p")
    os.makedirs(EOS_CACHE_DIR, exist_ok=True)

    fname = os.path.join(
        EOS_CACHE_DIR,
        f"ll_product_c_{safe_c}_nq{NQ_EOS}_nt{NTHETA_EOS}_nmax{N_TABLE_MAX:.4g}.npz",
    )

    if (not REBUILD_EOS) and os.path.exists(fname):
        print(f"Loading cached EOS: {fname}")
        data = np.load(fname)
        n_eos_np = data["n_eos"]
        e_eos_np = data["e_eos"]
        qs = data["qs"]
        qmax = float(data["qmax"])
    else:
        print("Building robust product-integration Bethe EOS...")
        n_eos_np, e_eos_np, qs, qmax = build_ll_eos_product(
            c=c,
            n0=n0,
            n_table_max=N_TABLE_MAX,
            n_q=NQ_EOS,
            n_theta=NTHETA_EOS,
        )

        np.savez(
            fname,
            n_eos=n_eos_np,
            e_eos=e_eos_np,
            qs=qs,
            qmax=qmax,
            c=c,
            n0=n0,
            n_table_max=N_TABLE_MAX,
        )
        print(f"Saved EOS cache: {fname}")

    # interpolation used by the torch action.
    e_pchip = scipy.interpolate.PchipInterpolator(n_eos_np, e_eos_np)

    # local derivative extraction.
    e0_np, mu0_np, e2_np, nn_fit, ee_fit, coeff, scale = local_poly_eos_derivatives(
        n_eos_np,
        e_eos_np,
        n0,
        window_frac=0.35,
        deg=5,
    )

    v_s_np = math.sqrt(max(n0 * e2_np, 1.0e-300))
    tau_h_seed_np = R / v_s_np
    S_tonks_np = 0.5 * (np.pi * n0 * R) ** 2

    print("\n================================================")
    print("EOS summary")
    print("================================================")
    print(f"c                      = {c:.10g}")
    print(f"n0                     = {n0:.10g}")
    print(f"EOS density range      = {n_eos_np[0]:.6g} to {n_eos_np[-1]:.6g}")
    print(f"number of EOS points   = {len(n_eos_np)}")
    print(f"e(n0)                  = {e0_np:.12g}")
    print(f"mu(n0)=e'(n0)          = {mu0_np:.12g}")
    print(f"e''(n0)                = {e2_np:.12g}")
    print(f"v_s                    = {v_s_np:.12g}")
    print(f"tau_h seed = R/v_s     = {tau_h_seed_np:.12g}")
    print(f"Tonks reference S_full = {S_tonks_np:.12g}")
    print(f"cache file             = {fname}")

    print("\n================================================")
    print("Weak-coupling sanity checks")
    print("================================================")

    weak_e0 = 0.5 * c * n0**2
    weak_mu0 = c * n0
    weak_vs = math.sqrt(c * n0)

    print(f"weak e0 = c n0^2/2     = {weak_e0:.12g}")
    print(f"ratio e0 / weak_e0     = {e0_np / weak_e0:.12g}")

    print(f"weak mu0 = c n0        = {weak_mu0:.12g}")
    print(f"ratio mu0 / weak_mu0   = {mu0_np / weak_mu0:.12g}")

    print(f"weak v_s = sqrt(c n0)  = {weak_vs:.12g}")
    print(f"ratio v_s / weak_v_s   = {v_s_np / weak_vs:.12g}")

    print("\nFor c << 1, these ratios should approach 1.")

    # check against finite differences
    dn_fd = 1.0e-3 * n0
    if n_eos_np[0] <= n0 - dn_fd and n0 + dn_fd <= n_eos_np[-1]:
        mu_fd = float((e_pchip(n0 + dn_fd) - e_pchip(n0 - dn_fd)) / (2.0 * dn_fd))
        e2_fd = float(
            (e_pchip(n0 + dn_fd) - 2.0 * e_pchip(n0) + e_pchip(n0 - dn_fd))
            / dn_fd**2
        )
        vs_fd = math.sqrt(max(n0 * e2_fd, 1.0e-300))

        print("\nDerivative cross-check:")
        print(f"mu0 local poly         = {mu0_np:.12g}")
        print(f"mu0 PCHIP finite diff  = {mu_fd:.12g}")
        print(f"v_s local poly         = {v_s_np:.12g}")
        print(f"v_s PCHIP finite diff  = {vs_fd:.12g}")

    # monotonicity checks.
    dn = np.diff(n_eos_np)
    de = np.diff(e_eos_np)

    print("\nNumerical sanity checks:")
    print(f"min diff(n)            = {np.min(dn):.4e}")
    print(f"min diff(e)            = {np.min(de):.4e}")
    print(f"EOS monotone in n?      = {np.all(dn > 0)}")
    print(f"EOS monotone in e?      = {np.all(de >= -1e-12)}")
    print(f"n0 inside table?        = {n_eos_np[0] <= n0 <= n_eos_np[-1]}")
    print(f"N_TABLE_MAX reached?    = {n_eos_np[-1] >= N_TABLE_MAX}")

    eos = {
        "n_eos_np": n_eos_np,
        "e_eos_np": e_eos_np,
        "e_pchip": e_pchip,
        "e0_np": float(e0_np),
        "mu0_np": float(mu0_np),
        "e2_np": float(e2_np),
        "v_s_np": float(v_s_np),
        "tau_h_seed_np": float(tau_h_seed_np),
        "S_tonks_np": float(S_tonks_np),
        "qs": qs,
        "qmax": qmax,
        "cache_file": fname,
    }

    return eos


def torch_interp_1d(x, xp, fp):
    """
    Piecewise-linear interpolation fp(xp) at tensor x.
    Differentiable with respect to x inside each interval.
    """

    x_clamped = torch.clamp(x, xp[0], xp[-1])
    flat = x_clamped.reshape(-1)

    idx = torch.searchsorted(xp, flat, right=False) - 1
    idx = torch.clamp(idx, 0, xp.numel() - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    t = (flat - x0) / (x1 - x0)
    out = y0 + t * (y1 - y0)

    return out.reshape_as(x)


# EOS-dependent globals are initialized in ``initialize_eos``.
eos = None
n_eos = None
e_eos = None
e0 = None
mu0 = None
v_s_np = None
tau_h_seed_np = None
S_tonks_np = None
tau_h_seed = None
log_tau_h_seed = None


def initialize_eos():
    """Build or load the EOS and initialize tensors used by the solver."""
    global eos, n_eos, e_eos, e0, mu0
    global v_s_np, tau_h_seed_np, S_tonks_np
    global tau_h_seed, log_tau_h_seed

    eos = load_or_build_eos()

    n_eos = torch.tensor(eos["n_eos_np"], device=device)
    e_eos = torch.tensor(eos["e_eos_np"], device=device)
    e0 = torch.tensor(eos["e0_np"], device=device)
    mu0 = torch.tensor(eos["mu0_np"], device=device)

    v_s_np = eos["v_s_np"]
    tau_h_seed_np = eos["tau_h_seed_np"]
    S_tonks_np = eos["S_tonks_np"]

    tau_h_seed = torch.tensor(tau_h_seed_np, device=device)
    log_tau_h_seed = torch.tensor(math.log(tau_h_seed_np), device=device)


def eos_e_torch(n):
    """Evaluate the tabulated EOS with differentiable linear interpolation."""
    if n_eos is None or e_eos is None:
        raise RuntimeError("EOS is not initialized. Call initialize_eos() first.")
    return torch_interp_1d(n, n_eos, e_eos)


def plot_eos_diagnostics():
    """Plot the EOS table and excess energy density."""
    if eos is None:
        raise RuntimeError("EOS is not initialized. Call initialize_eos() first.")

    plt.figure(figsize=(6, 4))
    plt.plot(eos["n_eos_np"], eos["e_eos_np"], ".", markersize=2, label="EOS table")
    n_plot = np.linspace(0.0, min(N_TABLE_MAX, max(3.0 * n0, 1.5 * n0)), 500)
    plt.plot(n_plot, eos["e_pchip"](n_plot), label="PCHIP")
    plt.axvline(n0, linestyle="--", label=r"$n_0$")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$e(n)$")
    plt.title("Lieb-Liniger EOS")
    plt.legend()
    plt.tight_layout()
    plt.show()

    h_plot = eos["e_pchip"](n_plot) - eos["e0_np"] - eos["mu0_np"] * (n_plot - n0)

    plt.figure(figsize=(6, 4))
    plt.plot(n_plot, h_plot)
    plt.axvline(n0, linestyle="--", label=r"$n_0$")
    plt.axhline(0.0, linestyle=":")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$h(n)$")
    plt.title("Excess energy density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================================================================
# INTERPOLATION HELPER (not used)
# ================================================================

def interpolate_phi_to_grid(phi_old, r_old, y_old, r_new, y_new):
    """
    Interpolate phi_old(r,y) to new tensor grid.
    Uses scipy RegularGridInterpolator on CPU.
    """

    interp = scipy.interpolate.RegularGridInterpolator(
        (r_old, y_old),
        phi_old,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    Rg, Yg = np.meshgrid(r_new, y_new, indexing="ij")
    pts = np.stack([Rg.reshape(-1), Yg.reshape(-1)], axis=1)

    phi_new = interp(pts).reshape(len(r_new), len(y_new))
    phi_new = np.nan_to_num(phi_new, nan=0.0, posinf=0.0, neginf=0.0)

    return phi_new


# ================================================================
# MAIN SOLVER CLASS
# ================================================================

class EFPSolver:
    def __init__(self, NR, NY, init_state=None):
        self.NR = int(NR)
        self.NY = int(NY)

        self.alpha_x = 2.0
        self.alpha_tau = 3.0

        self.build_grid()

        if init_state is not None:
            shape_init = init_state["shape_coeff_np"]
            log_tau_init = init_state["log_tau_h_np"]
        else:
            shape_init = np.zeros(N_SHAPE_MODES)
            log_tau_init = math.log(tau_h_seed_np)

        self.shape_coeff = torch.nn.Parameter(
            torch.tensor(shape_init, device=device, dtype=torch.float64)
        )

        self.log_tau_h = torch.nn.Parameter(
            torch.tensor(log_tau_init, device=device, dtype=torch.float64)
        )

        if init_state is None:
            phi_init = self.make_initial_phi().detach().cpu().numpy()
        else:
            phi_init = interpolate_phi_to_grid(
                init_state["phi_np"],
                init_state["r_nodes_np"],
                init_state["y_mid_np"],
                self.r_nodes.detach().cpu().numpy(),
                self.y_mid.detach().cpu().numpy(),
            )

        self.phi = torch.nn.Parameter(
            torch.tensor(phi_init, device=device, dtype=torch.float64)
        )

        self.history = []

    def build_grid(self):
        rmax = TAU_FACTOR ** (1.0 / self.alpha_tau)

        self.N_HOLE = int(0.72 * self.NR)
        self.N_TAIL = self.NR - self.N_HOLE + 1

        r_left = torch.linspace(0.0, 1.0, self.N_HOLE, device=device)
        r_right = torch.linspace(1.0, rmax, self.N_TAIL, device=device)[1:]
        r_nodes = torch.cat([r_left, r_right], dim=0)

        self.r_left = r_left
        self.r_nodes = r_nodes
        self.NR_actual = r_nodes.numel()

        self.dr_edges = r_nodes[1:] - r_nodes[:-1]
        self.r_mid = 0.5 * (r_nodes[1:] + r_nodes[:-1])

        self.dy = YMAX / self.NY
        self.y_mid = (torch.arange(self.NY, device=device) + 0.5) * self.dy

        self.env_y = torch.clamp((YMAX - self.y_mid) / YMAX, min=0.0) ** 2

        transition_width = 3.0 / max(self.N_HOLE, 2)
        self.hole_switch = torch.sigmoid(
            (1.0 - self.r_nodes) / transition_width
        )[:, None]

        ell_hole = 0.12 * R
        hole_shape_y = torch.exp(-(self.y_mid[None, :] / ell_hole) ** 2)

        self.boundary_factor = 1.0 - self.hole_switch * hole_shape_y
        self.boundary_factor = torch.clamp(self.boundary_factor, min=1.0e-5)

        hole_interval_mask = self.r_mid <= 1.0 + 1.0e-12
        self.s_mid_hole = self.r_mid[hole_interval_mask]
        self.ds_hole = self.dr_edges[hole_interval_mask]

        z_hole = 2.0 * self.s_mid_hole - 1.0
        self.B_hole = self.chebyshev_basis(z_hole, N_SHAPE_MODES)

        self.degree_weights = torch.arange(
            1,
            N_SHAPE_MODES + 1,
            device=device,
            dtype=torch.float64,
        )

        bump_center = 0.45 * R
        bump_width = 0.25 * R
        self.bump_y = torch.exp(-((self.y_mid - bump_center) / bump_width) ** 2)
        self.int_bump = torch.sum(self.bump_y) * self.dy

    def chebyshev_basis(self, z, M):
        if M == 0:
            return torch.empty((z.numel(), 0), device=device, dtype=torch.float64)

        cols = []
        T0 = torch.ones_like(z)
        T1 = z

        if M >= 1:
            cols.append(T1)

        Tm2 = T0
        Tm1 = T1

        for _m in range(2, M + 1):
            Tm = 2.0 * z * Tm1 - Tm2
            cols.append(Tm)
            Tm2 = Tm1
            Tm1 = Tm

        return torch.stack(cols, dim=1)

    def compute_X_nodes(self):
        s = self.s_mid_hole
        ds = self.ds_hole

        base = (s + 1.0e-12) ** (self.alpha_x - 1.0)
        base = base * torch.sqrt(torch.clamp(1.0 - s ** self.alpha_x, min=1.0e-12))

        if N_SHAPE_MODES > 0:
            eta_raw = self.B_hole @ self.shape_coeff
            eta = ETA_MAX * torch.tanh(eta_raw / ETA_MAX)
        else:
            eta = torch.zeros_like(s)

        speed = base * torch.exp(eta)

        increments = speed * ds
        total = torch.sum(increments) + 1.0e-30

        cdf_hole = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.float64),
                torch.cumsum(increments, dim=0) / total,
            ],
            dim=0,
        )

        X_hole = R * (1.0 - cdf_hole)

        if self.NR_actual > self.N_HOLE:
            X_tail = torch.zeros(
                self.NR_actual - self.N_HOLE,
                device=device,
                dtype=torch.float64,
            )
            X_nodes = torch.cat([X_hole, X_tail], dim=0)
        else:
            X_nodes = X_hole

        return X_nodes

    def compute_boundary_regularizers(self):
        X_nodes = self.compute_X_nodes()
        s = self.s_mid_hole

        if N_SHAPE_MODES > 0:
            eta_raw = self.B_hole @ self.shape_coeff
            eta = ETA_MAX * torch.tanh(eta_raw / ETA_MAX)

            shape_coeff_pen = torch.sum((self.degree_weights * self.shape_coeff) ** 2)
            shape_amp_pen = torch.mean(eta ** 2)

            if eta.numel() > 1:
                deta = eta[1:] - eta[:-1]
                ds = s[1:] - s[:-1]
                shape_slope_pen = torch.mean((deta / ds) ** 2)
            else:
                shape_slope_pen = torch.tensor(0.0, device=device, dtype=torch.float64)
        else:
            shape_coeff_pen = torch.tensor(0.0, device=device, dtype=torch.float64)
            shape_amp_pen = torch.tensor(0.0, device=device, dtype=torch.float64)
            shape_slope_pen = torch.tensor(0.0, device=device, dtype=torch.float64)

        X_hole = X_nodes[:self.N_HOLE]
        X_seed_hole = R * torch.clamp(1.0 - self.r_left ** self.alpha_x, min=0.0) ** 1.5

        X_seed_pen = torch.mean(((X_hole - X_seed_hole) / R) ** 2)

        dX = X_hole[1:] - X_hole[:-1]
        drh = self.r_left[1:] - self.r_left[:-1]
        Xr = dX / drh

        if Xr.numel() > 1:
            dXr = Xr[1:] - Xr[:-1]
            dr_mid_h = 0.5 * (drh[1:] + drh[:-1])
            X_curv_pen = torch.mean((dXr / dr_mid_h) ** 2)
        else:
            X_curv_pen = torch.tensor(0.0, device=device, dtype=torch.float64)

        shape_pen = (
            shape_coeff_pen
            + 2.0 * shape_amp_pen
            + 0.02 * shape_slope_pen
            + 1.0 * X_seed_pen
            + 0.002 * X_curv_pen
        )

        return {
            "shape_pen": shape_pen,
            "shape_coeff_pen": shape_coeff_pen,
            "shape_amp_pen": shape_amp_pen,
            "shape_slope_pen": shape_slope_pen,
            "X_seed_pen": X_seed_pen,
            "X_curv_pen": X_curv_pen,
        }

    def make_initial_phi(self):
        with torch.no_grad():
            X_nodes = self.compute_X_nodes()
            target_n = torch.empty(
                (self.NR_actual, self.NY),
                device=device,
                dtype=torch.float64,
            )

            for i in range(self.NR_actual):
                X = X_nodes[i]
                base = n0 * self.boundary_factor[i, :]

                base_defect = torch.sum(base - n0) * self.dy
                desired_defect = n0 * X

                amp = (desired_defect - base_defect) / self.int_bump
                amp = torch.clamp(amp, min=0.0, max=8.0 * n0)

                profile = base + amp * self.bump_y
                profile = torch.clamp(profile, min=1.0e-8, max=12.0 * n0)

                target_n[i, :] = profile

            denom = n0 * self.boundary_factor
            ratio = torch.clamp(target_n / denom, min=1.0e-8, max=1.0e8)

            phi0 = torch.log(ratio) / torch.clamp(self.env_y[None, :], min=0.08)

            for _ in range(5):
                phi0[1:-1, :] = (
                    0.25 * phi0[:-2, :]
                    + 0.50 * phi0[1:-1, :]
                    + 0.25 * phi0[2:, :]
                )

        return phi0

    def compute_fields(self):
        X_nodes = self.compute_X_nodes()

        tau_h = torch.exp(self.log_tau_h)
        tau_nodes = tau_h * self.r_nodes ** self.alpha_tau

        tau_r = (tau_nodes[1:] - tau_nodes[:-1]) / self.dr_edges
        X_r = (X_nodes[1:] - X_nodes[:-1]) / self.dr_edges

        exponent = self.env_y[None, :] * self.phi
        exponent = torch.clamp(exponent, min=EXPONENT_MIN, max=EXPONENT_MAX)

        n = n0 * self.boundary_factor * torch.exp(exponent) + 1.0e-12

        Q = self.dy * (torch.cumsum(n, dim=1) - 0.5 * n)

        Q_r = (Q[1:, :] - Q[:-1, :]) / self.dr_edges[:, None]
        n_mid = 0.5 * (n[1:, :] + n[:-1, :])

        A = Q_r - X_r[:, None] * n_mid
        u_mid = -A / (tau_r[:, None] * n_mid)

        X_mid = 0.5 * (X_nodes[1:] + X_nodes[:-1])
        tau_mid = 0.5 * (tau_nodes[1:] + tau_nodes[:-1])

        return {
            "X_nodes": X_nodes,
            "X_mid": X_mid,
            "tau_h": tau_h,
            "tau_nodes": tau_nodes,
            "tau_mid": tau_mid,
            "tau_r": tau_r,
            "X_r": X_r,
            "n": n,
            "Q": Q,
            "Q_r": Q_r,
            "n_mid": n_mid,
            "A": A,
            "u_mid": u_mid,
        }

    def loss_function(self):
        f = self.compute_fields()

        X_nodes = f["X_nodes"]
        X_mid = f["X_mid"]
        n = f["n"]
        n_mid = f["n_mid"]
        A = f["A"]
        u_mid = f["u_mid"]
        tau_r = f["tau_r"]

        e_mid = eos_e_torch(n_mid)
        h_mid = e_mid - e0 - mu0 * (n_mid - n0)

        kinetic = A * A / (2.0 * tau_r[:, None] * n_mid)
        potential_exterior = tau_r[:, None] * h_mid

        kinetic_quad = torch.sum(kinetic * self.dr_edges[:, None]) * self.dy
        potential_exterior_quad = torch.sum(potential_exterior * self.dr_edges[:, None]) * self.dy

        h_empty = mu0 * n0 - e0

        if INCLUDE_HOLE_ACTION:
            hole_interval_mask = self.r_mid <= 1.0 + 1.0e-12
            hole_action_quad = h_empty * torch.sum(
                tau_r[hole_interval_mask]
                * X_mid[hole_interval_mask]
                * self.dr_edges[hole_interval_mask]
            )
        else:
            hole_action_quad = torch.tensor(0.0, device=device, dtype=torch.float64)

        potential_quad = potential_exterior_quad + hole_action_quad

        action_quad = kinetic_quad + potential_quad
        S_full = 4.0 * action_quad

        # mass/displacement constraint
        mass_defect = torch.sum(n - n0, dim=1) * self.dy
        mass_target = n0 * X_nodes
        mass_pen = torch.mean((mass_defect - mass_target) ** 2)

        far_cols = min(6, self.NY)
        far_density_pen = torch.mean(((n[:, -far_cols:] - n0) / n0) ** 2)
        far_A_pen = torch.mean(A[:, -far_cols:] ** 2)

        start_sym_pen = torch.mean(A[0, :] ** 2)
        top_density_pen = torch.mean(((n[-1, :] - n0) / n0) ** 2)

        axis_node_mask = (self.r_nodes > 1.0).to(n.dtype)
        axis_interval_mask = (self.r_mid > 1.0).to(n.dtype)

        axis_density_pen = torch.sum(
            axis_node_mask * ((n[:, 1] - n[:, 0]) / n0) ** 2
        ) / (torch.sum(axis_node_mask) + 1.0e-12)

        axis_u_cols = min(3, self.NY)
        axis_velocity_pen = torch.sum(
            axis_interval_mask[:, None] * u_mid[:, :axis_u_cols] ** 2
        ) / (torch.sum(axis_interval_mask) * axis_u_cols + 1.0e-12)

        hole_node_mask = (self.r_nodes < 1.0).to(n.dtype)
        hole_diag = torch.sum(
            hole_node_mask * (n[:, 0] / n0) ** 2
        ) / (torch.sum(hole_node_mask) + 1.0e-12)

        hole_edge_pen = hole_diag

        smooth_r = torch.mean((self.phi[1:, :] - self.phi[:-1, :]) ** 2)
        smooth_y = torch.mean((self.phi[:, 1:] - self.phi[:, :-1]) ** 2)

        bregs = self.compute_boundary_regularizers()
        shape_pen = bregs["shape_pen"]

        overflow_pen = torch.mean(torch.relu(n - DENSITY_OVERFLOW_CAP) ** 2)

        # penalty weights
        W_MASS = 250.0
        W_FAR_DENS = 40.0
        W_FAR_A = 20.0
        W_START = 25.0
        W_TOP = 40.0
        W_AXIS_DENS = 10.0
        W_AXIS_U = 10.0
        W_SMOOTH = 3.0e-4
        W_SHAPE = 2.0e-3
        W_OVERFLOW = 200.0

        # kept at zero
        W_HOLE_EDGE = 0

        penalties = (
            W_MASS * mass_pen
            + W_FAR_DENS * far_density_pen
            + W_FAR_A * far_A_pen
            + W_START * start_sym_pen
            + W_TOP * top_density_pen
            + W_AXIS_DENS * axis_density_pen
            + W_AXIS_U * axis_velocity_pen
            + W_SMOOTH * (smooth_r + smooth_y)
            + W_SHAPE * shape_pen
            + W_OVERFLOW * overflow_pen
            + W_HOLE_EDGE * hole_edge_pen
        )

        loss = action_quad + penalties
        virial_ratio = kinetic_quad / (potential_quad + 1.0e-30)

        diagnostics = {
            "loss": loss.detach(),
            "action_quad": action_quad.detach(),
            "S_full": S_full.detach(),
            "penalty_total": penalties.detach(),
            "kinetic_quad": kinetic_quad.detach(),
            "potential_quad": potential_quad.detach(),
            "potential_exterior_quad": potential_exterior_quad.detach(),
            "hole_action_quad": hole_action_quad.detach(),
            "h_empty": h_empty.detach(),
            "virial_ratio": virial_ratio.detach(),
            "mass_pen": mass_pen.detach(),
            "far_density_pen": far_density_pen.detach(),
            "far_A_pen": far_A_pen.detach(),
            "start_sym_pen": start_sym_pen.detach(),
            "top_density_pen": top_density_pen.detach(),
            "axis_density_pen": axis_density_pen.detach(),
            "axis_velocity_pen": axis_velocity_pen.detach(),
            "hole_diag": hole_diag.detach(),
            "hole_edge_pen": hole_edge_pen.detach(),
            "smooth_r": smooth_r.detach(),
            "smooth_y": smooth_y.detach(),
            "shape_pen": shape_pen.detach(),
            "shape_coeff_pen": bregs["shape_coeff_pen"].detach(),
            "shape_amp_pen": bregs["shape_amp_pen"].detach(),
            "shape_slope_pen": bregs["shape_slope_pen"].detach(),
            "X_seed_pen": bregs["X_seed_pen"].detach(),
            "X_curv_pen": bregs["X_curv_pen"].detach(),
            "overflow_pen": overflow_pen.detach(),
            "tau_h": f["tau_h"].detach(),
        }

        return loss, diagnostics

    def snapshot_state(self):
        return {
            "phi": self.phi.detach().clone(),
            "shape_coeff": self.shape_coeff.detach().clone(),
            "log_tau_h": self.log_tau_h.detach().clone(),
        }

    def restore_state(self, state):
        with torch.no_grad():
            self.phi.copy_(state["phi"])
            self.shape_coeff.copy_(state["shape_coeff"])
            self.log_tau_h.copy_(state["log_tau_h"])

    def print_diag(self, prefix, it, diag):
        d = {k: float(v.detach().cpu()) for k, v in diag.items()}
        print(
            f"{prefix} it={it:5d} | "
            f"loss={d['loss']:.6e} | "
            f"S_full={d['S_full']:.6e} | "
            f"pen={d['penalty_total']:.3e} | "
            f"tau_h={d['tau_h']:.6e} | "
            f"K/V={d['virial_ratio']:.3e} | "
            f"mass={d['mass_pen']:.2e} | "
            f"hole={d['hole_diag']:.2e} | "
            f"shape={d['shape_pen']:.2e}"
        )

    def optimizer_parameter_list(self, include_tau=True):
        params = [self.phi]

        if N_SHAPE_MODES > 0:
            params.append(self.shape_coeff)

        if include_tau:
            params.append(self.log_tau_h)

        return params

    def run_adam(self, steps, label="Adam", optimize_tau=ADAM_OPTIMIZE_TAU):
        groups = [{"params": [self.phi], "lr": 3.0e-4}]

        if N_SHAPE_MODES > 0:
            groups.append({"params": [self.shape_coeff], "lr": 5.0e-5})

        if optimize_tau:
            groups.append({"params": [self.log_tau_h], "lr": ADAM_TAU_LR})

        optimizer = torch.optim.Adam(groups)

        best_loss = float("inf")
        best_state = self.snapshot_state()

        for it in range(steps + 1):
            optimizer.zero_grad()

            loss, diag = self.loss_function()

            loss_float = float(loss.detach().cpu())
            if loss_float < best_loss:
                best_loss = loss_float
                best_state = self.snapshot_state()

            loss.backward()

            torch.nn.utils.clip_grad_norm_([self.phi], max_norm=3.0)

            if N_SHAPE_MODES > 0:
                torch.nn.utils.clip_grad_norm_([self.shape_coeff], max_norm=0.5)

            if optimize_tau:
                torch.nn.utils.clip_grad_norm_([self.log_tau_h], max_norm=0.01)

            optimizer.step()

            if it % PRINT_EVERY == 0 or it == steps:
                self.print_diag(label, it, diag)
                self.history.append(
                    (label, it, {k: float(v.detach().cpu()) for k, v in diag.items()})
                )

        self.restore_state(best_state)

        _, diag = self.loss_function()
        self.print_diag(label + " restored", steps, diag)

    def run_lbfgs(self, max_iter, label="LBFGS"):
        params = self.optimizer_parameter_list(include_tau=True)

        lbfgs = torch.optim.LBFGS(
            params,
            lr=0.4,
            max_iter=max_iter,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-14,
            history_size=LBFGS_HISTORY_SIZE,
            line_search_fn="strong_wolfe",
        )

        calls = {"n": 0}
        best = {
            "loss": float("inf"),
            "state": self.snapshot_state(),
        }

        def closure():
            lbfgs.zero_grad()

            loss, diag = self.loss_function()

            loss_float = float(loss.detach().cpu())
            if loss_float < best["loss"]:
                best["loss"] = loss_float
                best["state"] = self.snapshot_state()

            loss.backward()

            calls["n"] += 1

            if calls["n"] % 50 == 0:
                self.print_diag(label, calls["n"], diag)
                self.history.append(
                    (label, calls["n"], {k: float(v.detach().cpu()) for k, v in diag.items()})
                )

            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError as err:
            print(f"{label} stopped with RuntimeError:")
            print(err)
            print("Continuing with best LBFGS state.")

        self.restore_state(best["state"])

        _, diag = self.loss_function()
        self.print_diag(label + " restored", max_iter, diag)

    def export_state(self):
        with torch.no_grad():
            return {
                "phi_np": self.phi.detach().cpu().numpy(),
                "r_nodes_np": self.r_nodes.detach().cpu().numpy(),
                "y_mid_np": self.y_mid.detach().cpu().numpy(),
                "shape_coeff_np": self.shape_coeff.detach().cpu().numpy(),
                "log_tau_h_np": float(self.log_tau_h.detach().cpu()),
                "history": self.history,
            }

    def final_diagnostics(self):
        loss, diag = self.loss_function()
        d = {k: float(v.detach().cpu()) for k, v in diag.items()}

        print("\nFinal diagnostics:")
        for key, val in d.items():
            print(f"{key:24s}: {val:.10e}")

        print("\nReference values:")
        print(f"Tonks target S_full       = {S_tonks_np:.10g}")
        print(f"Seed tau_h = R/v_s       = {tau_h_seed_np:.10g}")
        print(f"Optimized tau_h           = {d['tau_h']:.10g}")
        print(f"Scaled height v_s tau_h/R = {v_s_np * d['tau_h'] / R:.10g}")
        print(f"Virial ratio K/V          = {d['virial_ratio']:.10g}")
        print(f"Penalty/action_quad       = {d['penalty_total'] / (d['action_quad'] + 1e-30):.10g}")

        return d

    def extract_numpy_fields(self):
        with torch.no_grad():
            f = self.compute_fields()

            return {
                "X_nodes": f["X_nodes"].detach().cpu().numpy(),
                "X_mid": f["X_mid"].detach().cpu().numpy(),
                "tau_h": float(f["tau_h"].detach().cpu()),
                "tau_nodes": f["tau_nodes"].detach().cpu().numpy(),
                "tau_mid": f["tau_mid"].detach().cpu().numpy(),
                "r_nodes": self.r_nodes.detach().cpu().numpy(),
                "r_mid": self.r_mid.detach().cpu().numpy(),
                "y_mid": self.y_mid.detach().cpu().numpy(),
                "n_nodes": f["n"].detach().cpu().numpy(),
                "n_mid": f["n_mid"].detach().cpu().numpy(),
                "u_mid": f["u_mid"].detach().cpu().numpy(),
                "A": f["A"].detach().cpu().numpy(),
                "shape_coeff": self.shape_coeff.detach().cpu().numpy(),
            }


# ================================================================
# CONTINUATION
# ================================================================

def run_continuation():
    """Run the configured coarse-to-fine optimization and return final fields."""
    state = None
    solvers = []

    for level_id, (NR, NY, cycles, adam_steps, lbfgs_steps) in enumerate(LEVELS):
        print("\n" + "=" * 70)
        print(f"LEVEL {level_id + 1}/{len(LEVELS)}: NR={NR}, NY={NY}")
        print("=" * 70)

        solver = EFPSolver(NR=NR, NY=NY, init_state=state)

        for cycle in range(cycles):
            print("\n" + "-" * 60)
            print(f"Level {level_id + 1}, cycle {cycle + 1}/{cycles}")
            print("-" * 60)

            if adam_steps > 0:
                solver.run_adam(
                    steps=adam_steps,
                    label=f"L{level_id + 1}C{cycle + 1} Adam",
                    optimize_tau=ADAM_OPTIMIZE_TAU,
                )

            for restart in range(LBFGS_RESTARTS_PER_CYCLE):
                solver.run_lbfgs(
                    max_iter=lbfgs_steps,
                    label=f"L{level_id + 1}C{cycle + 1} LBFGS-{restart + 1}",
                )

        solver.final_diagnostics()

        state = solver.export_state()
        solvers.append(solver)

    final_solver = solvers[-1]
    fields = final_solver.extract_numpy_fields()

    return solvers, fields


# ================================================================
# PLOT HELPERS
# ================================================================

def boundary_polygon_from_solution(fields):
    r_nodes_np = fields["r_nodes"]
    X_nodes_np = fields["X_nodes"]
    tau_nodes_np = fields["tau_nodes"]

    hole_nodes = r_nodes_np <= 1.0 + 1.0e-12
    xb = X_nodes_np[hole_nodes]
    tb = tau_nodes_np[hole_nodes]

    upper_x = np.r_[xb, -xb[::-1]]
    upper_t = np.r_[tb, tb[::-1]]

    lower_x = upper_x[::-1]
    lower_t = -upper_t[::-1]

    poly_x = np.r_[upper_x, lower_x]
    poly_t = np.r_[upper_t, lower_t]

    return poly_x, poly_t


def x_boundary_at_tau_abs(t_abs, fields):
    r_nodes_np = fields["r_nodes"]
    X_nodes_np = fields["X_nodes"]
    tau_nodes_np = fields["tau_nodes"]

    hole_nodes = r_nodes_np <= 1.0 + 1.0e-12
    tb = tau_nodes_np[hole_nodes]
    xb = X_nodes_np[hole_nodes]

    t_clipped = np.clip(t_abs, tb[0], tb[-1])
    return np.interp(t_clipped, tb, xb)


def mask_triangles_inside_hole(tri, x, tau, fields):
    triangles = tri.triangles
    xc = x[triangles].mean(axis=1)
    tc = tau[triangles].mean(axis=1)

    at = np.abs(tc)
    xb = x_boundary_at_tau_abs(at, fields)

    inside = (at <= fields["tau_h"]) & (np.abs(xc) < xb)
    return inside


def make_full_reflected_points(field_quad, fields, kind="density", max_plot_points_per_dim=250):
    X_mid_np = fields["X_mid"]
    tau_mid_np = fields["tau_mid"]
    y_mid_np = fields["y_mid"]

    nr = field_quad.shape[0]
    ny = field_quad.shape[1]

    sr = max(1, nr // max_plot_points_per_dim)
    sy = max(1, ny // max_plot_points_per_dim)

    field_sub = field_quad[::sr, ::sy]
    X_sub = X_mid_np[::sr]
    tau_sub = tau_mid_np[::sr]
    y_sub = y_mid_np[::sy]

    xq = X_sub[:, None] + y_sub[None, :]
    tq = tau_sub[:, None] + np.zeros_like(xq)

    xs = []
    ts = []
    vals = []

    for sx in [+1.0, -1.0]:
        for st in [+1.0, -1.0]:
            x = sx * xq
            t = st * tq

            if kind == "density":
                val = field_sub
            elif kind == "velocity":
                val = sx * st * field_sub
            else:
                raise ValueError("kind must be density or velocity")

            xs.append(x.reshape(-1))
            ts.append(t.reshape(-1))
            vals.append(val.reshape(-1))

    return np.concatenate(xs), np.concatenate(ts), np.concatenate(vals)


def plot_full_field(field_quad, fields, kind, title, label, levels=90):
    x, t, vals = make_full_reflected_points(field_quad, fields, kind=kind)

    tri = mtri.Triangulation(x, t)
    tri.set_mask(mask_triangles_inside_hole(tri, x, t, fields))

    plt.figure(figsize=(7.4, 6.0))
    cf = plt.tricontourf(tri, vals, levels=levels)
    plt.colorbar(cf, label=label)

    bx, bt = boundary_polygon_from_solution(fields)
    plt.plot(bx, bt, linewidth=2.0)
    plt.fill(bx, bt, alpha=0.14)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$\tau$")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# ================================================================
# FINAL PLOTTERS
# ================================================================

def plot_final_results(fields, solvers):
    """Generate the final field, boundary, and optimization-history plots."""
    plot_full_field(
        fields["n_mid"],
        fields,
        kind="density",
        title=rf"EFP density $n(x,\tau)$, c={c}",
        label=r"$n$",
    )

    plot_full_field(
        fields["u_mid"],
        fields,
        kind="velocity",
        title=rf"Imaginary-time velocity $u(x,\tau)$, c={c}",
        label=r"$u$",
    )

    # Cross-section near tau=0.
    X_mid_np = fields["X_mid"]
    y_mid_np = fields["y_mid"]
    n_mid_np = fields["n_mid"]

    x_right = X_mid_np[0] + y_mid_np
    n_right = n_mid_np[0, :]

    x_hole = np.linspace(-R, R, 300)
    n_hole = np.zeros_like(x_hole)

    x_cross = np.r_[-x_right[::-1], x_hole, x_right]
    n_cross = np.r_[n_right[::-1], n_hole, n_right]

    plt.figure(figsize=(7, 4))
    plt.plot(x_cross, n_cross)
    plt.axvline(-R, linestyle="--")
    plt.axvline(R, linestyle="--")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n(x,\tau\approx 0)$")
    plt.title(r"Density cross-section near $\tau=0$")
    plt.tight_layout()
    plt.show()

    # Boundary in physical time.
    bx, bt = boundary_polygon_from_solution(fields)

    plt.figure(figsize=(6, 5))
    plt.plot(bx, bt)
    plt.fill(bx, bt, alpha=0.14)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\tau$")
    plt.title("Optimized empty-region boundary")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Boundary in scaled time.
    plt.figure(figsize=(6, 5))
    plt.plot(bx / R, v_s_np * bt / R)
    plt.fill(bx / R, v_s_np * bt / R, alpha=0.14)
    plt.xlabel(r"$x/R$")
    plt.ylabel(r"$v_s\tau/R$")
    plt.title("Optimized boundary in scaled time")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Boundary profile versus seed.
    r_nodes_np = fields["r_nodes"]
    X_nodes_np = fields["X_nodes"]
    tau_nodes_np = fields["tau_nodes"]

    hole_nodes = r_nodes_np <= 1.0 + 1.0e-12

    rr_seed = np.linspace(0.0, 1.0, 400)
    tau_seed = tau_h_seed_np * rr_seed ** 3
    X_seed = R * np.maximum(1.0 - rr_seed ** 2, 0.0) ** 1.5

    plt.figure(figsize=(6, 4))
    plt.plot(tau_nodes_np[hole_nodes], X_nodes_np[hole_nodes], label="optimized")
    plt.plot(tau_seed, X_seed, linestyle="--", label="seed")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$x_b(\tau)$")
    plt.title("Right boundary profile")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(r_nodes_np[hole_nodes], X_nodes_np[hole_nodes], label="optimized")
    plt.plot(rr_seed, X_seed, linestyle="--", label="seed")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$X(r)$")
    plt.title("Boundary profile in computational coordinate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # History.
    all_history = []
    for s in solvers:
        all_history.extend(s.history)

    if all_history:
        its = np.arange(len(all_history))
        Sfull_hist = np.array([h[2]["S_full"] for h in all_history])
        loss_hist = np.array([h[2]["loss"] for h in all_history])
        penalty_hist = np.array([h[2]["penalty_total"] for h in all_history])
        tau_hist = np.array([h[2]["tau_h"] for h in all_history])
        virial_hist = np.array([h[2]["virial_ratio"] for h in all_history])
        mass_hist = np.array([h[2]["mass_pen"] for h in all_history])
        hole_hist = np.array([h[2]["hole_diag"] for h in all_history])
        shape_hist = np.array([h[2]["shape_pen"] for h in all_history])

        plt.figure(figsize=(7, 4))
        plt.semilogy(its, Sfull_hist, marker="o", markersize=3, label="S_full")
        plt.axhline(S_tonks_np, linestyle="--", label="Tonks target")
        plt.xlabel("logged checkpoint")
        plt.ylabel(r"$S_{\rm full}$")
        plt.title("Action history")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.semilogy(its, loss_hist, marker="o", markersize=3, label="loss")
        plt.semilogy(its, penalty_hist, marker="o", markersize=3, label="penalty")
        plt.xlabel("logged checkpoint")
        plt.ylabel("value")
        plt.title("Loss and penalty history")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(its, tau_hist, marker="o", markersize=3)
        plt.axhline(tau_h_seed_np, linestyle="--", label="seed R/v_s")
        plt.xlabel("logged checkpoint")
        plt.ylabel(r"$\tau_h$")
        plt.title(r"$\tau_h$ history")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.semilogy(its, virial_hist, marker="o", markersize=3)
        plt.axhline(1.0, linestyle="--", label="K/V = 1")
        plt.xlabel("logged checkpoint")
        plt.ylabel(r"$S_{\rm kin}/S_{\rm pot}$")
        plt.title("Virial balance history")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.semilogy(its, mass_hist, label="mass")
        plt.semilogy(its, hole_hist, label="hole edge diagnostic")
        plt.semilogy(its, shape_hist, label="shape")
        plt.xlabel("logged checkpoint")
        plt.ylabel("diagnostic")
        plt.title("Constraint diagnostics")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\nFinal field ranges:")
    print(f"min n = {np.min(fields['n_mid']):.6e}")
    print(f"max n = {np.max(fields['n_mid']):.6e}")
    print(f"min u = {np.min(fields['u_mid']):.6e}")
    print(f"max u = {np.max(fields['u_mid']):.6e}")
    print(f"optimized tau_h = {fields['tau_h']:.10g}")
    print(f"scaled height v_s*tau_h/R = {v_s_np * fields['tau_h'] / R:.10g}")

    if np.max(fields["n_mid"]) > 0.85 * N_TABLE_MAX:
        print("\nWARNING: density is approaching EOS table maximum.")
        print("Increase N_TABLE_MAX and rebuild the EOS table.")


def main():
    """Run the supplementary calculation with the configuration above."""
    print(f"Device: {device}")
    initialize_eos()
    plot_eos_diagnostics()

    solvers, fields = run_continuation()

    print("\nOptimized boundary coefficients:")
    print(fields["shape_coeff"])

    plot_final_results(fields, solvers)


if __name__ == "__main__":
    main()
