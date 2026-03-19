"""
Fig. 2: SAFE-GRAPE pulse fidelity and Bloch sphere.
Source: SAFE_GRAPE.ipynb in Error-Correcting Pulses_
Interactive: infidelity heatmap over (ε, f), Bloch trajectory for selected (ε, f).
Supports both rCinBB and SAFE-GRAPE sequences; pulse diagrams for both.
"""
import functools
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

# Paper/notebook parameters
OMEGA = 200e6  # Mrad/s
TIME_STEPS = 50  # Reduced for interactive speed (notebook uses 100)
THETA_TARGET = np.pi
PHI_TARGET = 0.0

# Cache path for pre-computed SAFE-GRAPE pulse (run scripts/generate_safe_grape_cache.py)
_SAFE_GRAPE_CACHE = Path(__file__).resolve().parent.parent / "data" / "safe_grape_pulse.npz"

# Pauli matrices
_sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
_sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def _safe_grape_available() -> bool:
    """Check if SAFE-GRAPE cache exists."""
    return _SAFE_GRAPE_CACHE.exists()


def _get_rCinBB_params():
    """rCinBB pulse parameters (reduced CORPSE in BB1). From SAFE_GRAPE.ipynb."""
    k = np.arcsin(np.sin(THETA_TARGET / 2) / 2)
    theta_ig = [
        np.pi,
        2 * np.pi,
        np.pi,
        2 * np.pi + THETA_TARGET / 2 - k,
        2 * np.pi - 2 * k,
        THETA_TARGET / 2 - k,
    ]
    phi_ig = [
        PHI_TARGET + np.arccos(-THETA_TARGET / 4 / np.pi),
        PHI_TARGET + 3 * np.arccos(-THETA_TARGET / 4 / np.pi),
        PHI_TARGET + np.arccos(-THETA_TARGET / 4 / np.pi),
        PHI_TARGET,
        PHI_TARGET + np.pi,
        PHI_TARGET,
    ]
    total_theta = sum(theta_ig)
    total_time = total_theta / OMEGA
    t_arr = np.zeros(TIME_STEPS)
    phi_arr = np.zeros(TIME_STEPS)
    for i in range(6):
        start_idx = int(np.rint(sum(theta_ig[:i]) / total_theta * TIME_STEPS))
        end_idx = int(np.rint(sum(theta_ig[:i + 1]) / total_theta * TIME_STEPS))
        t_arr[start_idx:end_idx] = total_time / TIME_STEPS
        phi_arr[start_idx:end_idx] = phi_ig[i]
    return t_arr, phi_arr


def _get_safe_grape_params():
    """
    Load SAFE-GRAPE pulse from cache. Returns (t_arr, phi_arr) compatible with
    compute_fidelity_map and _evolve_trajectory. Raises FileNotFoundError if cache missing.
    """
    data = np.load(_SAFE_GRAPE_CACHE)
    I = data["I"]
    Q = data["Q"]
    delta_t = float(data["delta_t"])
    omega_max = float(data["omega_max"])
    # Convert I,Q to (t, phi) format: t = delta_t * amp, phi = atan2(Q,I)
    amp = np.sqrt(I**2 + Q**2) + 1e-12
    t_arr = delta_t * amp
    phi_arr = np.arctan2(Q, I)
    # Scale so effective Omega matches our OMEGA (200 Mrad/s)
    # Real rotation per step: omega_max * delta_t * amp. We use Omega * t.
    # So t_eff = omega_max * delta_t * amp / OMEGA
    scale = omega_max / OMEGA
    t_arr = t_arr * scale
    return t_arr, phi_arr


def _get_pulse_params(sequence: str):
    """Get (t_arr, phi_arr) for given sequence. Raises if SAFE-GRAPE requested but unavailable."""
    if sequence == "rCinBB":
        return _get_rCinBB_params()
    if sequence == "SAFE-GRAPE":
        if not _safe_grape_available():
            raise FileNotFoundError(
                "SAFE-GRAPE cache not found. Run: python scripts/generate_safe_grape_cache.py"
            )
        return _get_safe_grape_params()
    raise ValueError(f"Unknown sequence: {sequence}")


def _V(t: float, phi: float, f: float, eps: float) -> np.ndarray:
    """Real gate with amplitude error eps and detuning f."""
    H = (1 + eps) * (np.cos(phi) * _sigma_x + np.sin(phi) * _sigma_y) + f * _sigma_z
    return scipy.linalg.expm(-1j * OMEGA * t / 2 * H)


def _U(theta: float, phi: float) -> np.ndarray:
    """Ideal gate."""
    H = np.cos(phi) * _sigma_x + np.sin(phi) * _sigma_y
    return scipy.linalg.expm(-1j * theta / 2 * H)


def _avg_gate_fidelity(U_mat: np.ndarray, V_mat: np.ndarray) -> float:
    """Average gate fidelity."""
    return np.abs(np.trace(U_mat.conj().T @ V_mat) / 2)


def compute_fidelity_map(
    t_arr: np.ndarray,
    phi_arr: np.ndarray,
    eps_range: float,
    f_range: float,
    n_eps: int,
    n_f: int,
) -> np.ndarray:
    """Compute fidelity over (epsilon, f) grid. Returns 2D array [eps_idx, f_idx]."""
    fid = np.zeros((n_eps, n_f))
    eps_vals = np.linspace(-eps_range, eps_range, n_eps)
    f_vals = np.linspace(-f_range, f_range, n_f)
    U_ideal = _U(THETA_TARGET, PHI_TARGET)
    for i, eps in enumerate(eps_vals):
        for j, f in enumerate(f_vals):
            seq = _V(0, 0, 0, 0)
            for n in range(len(t_arr)):
                seq = _V(t_arr[n], phi_arr[n], f, eps) @ seq
            fid[i, j] = _avg_gate_fidelity(seq, U_ideal)
    return fid


def _bloch_vector(psi: np.ndarray) -> tuple[float, float, float]:
    """Bloch vector (x,y,z) from state psi = [a,b]."""
    a, b = psi[0, 0], psi[1, 0]
    x = 2 * np.real(a * np.conj(b))
    y = 2 * np.imag(a * np.conj(b))
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return x, y, z


def _evolve_trajectory(
    t_arr: np.ndarray,
    phi_arr: np.ndarray,
    eps: float,
    f: float,
    n_pts_per_seg: int = 4,
) -> np.ndarray:
    """Evolve |0⟩ through pulse, return Bloch trajectory (Nx3)."""
    psi = np.array([[1], [0]], dtype=complex)
    traj = [_bloch_vector(psi)]
    for idx in range(len(t_arr)):
        H = OMEGA / 2 * (
            (1 + eps) * (np.cos(phi_arr[idx]) * _sigma_x + np.sin(phi_arr[idx]) * _sigma_y)
            + f * _sigma_z
        )
        for k in range(1, n_pts_per_seg + 1):
            t_frac = k / n_pts_per_seg
            U_part = scipy.linalg.expm(-1j * H * t_arr[idx] * t_frac)
            psi_step = U_part @ psi
            traj.append(_bloch_vector(psi_step))
        psi = scipy.linalg.expm(-1j * H * t_arr[idx]) @ psi
    return np.array(traj)


@functools.lru_cache(maxsize=8)
def _fidelity_map_cached(sequence: str, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached fidelity map. Returns (infid, eps_vals, f_vals). Heatmaps only depend on n_grid, not ε/f."""
    t_arr, phi_arr = _get_pulse_params(sequence)
    fid = compute_fidelity_map(t_arr, phi_arr, 0.3, 0.3, n_grid, n_grid)
    infid = 1 - fid
    eps_vals = np.linspace(-0.3, 0.3, n_grid)
    f_vals = np.linspace(-0.3, 0.3, n_grid)
    return (infid, eps_vals, f_vals)


def fig2_infidelity_heatmap(
    eps_range: float = 0.3,
    f_range: float = 0.3,
    n_grid: int = 25,
    sequence: str = "rCinBB",
) -> plt.Figure:
    """
    Interactive Fig. 2(b): Infidelity heatmap over (ε, f).
    sequence: 'rCinBB' or 'SAFE-GRAPE' (requires cache from generate_safe_grape_cache.py).
    """
    infid, eps_vals, f_vals = _fidelity_map_cached(sequence, n_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    infid_safe = np.clip(infid, 1e-6, 1)
    norm = LogNorm(vmin=infid_safe.min(), vmax=infid_safe.max())
    im = ax.imshow(infid, origin="lower", cmap="plasma", norm=norm, aspect="auto")
    levels = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    levels = [L for L in levels if infid.min() < L < infid.max()]
    if levels:
        ax.contour(infid, levels=levels, colors="white", linewidths=0.5)
    ax.set_xticks([0, n_grid // 2, n_grid - 1])
    ax.set_yticks([0, n_grid // 2, n_grid - 1])
    ax.set_xticklabels([f"{f_vals[0]:.2f}", "0", f"{f_vals[-1]:.2f}"])
    ax.set_yticklabels([f"{eps_vals[0]:.2f}", "0", f"{eps_vals[-1]:.2f}"])
    ax.set_xlabel(r"$f$ (detuning)")
    ax.set_ylabel(r"$\epsilon$ (amplitude error)")
    ax.set_title(f"Gate infidelity — {sequence} [Interactive]")
    plt.colorbar(im, ax=ax, label="Infidelity")
    fig.tight_layout()
    return fig


def fig2_bloch(
    epsilon: float = 0.25,
    f: float = 0.25,
    sequence: str = "rCinBB",
) -> plt.Figure:
    """
    Interactive Fig. 2(c): Bloch sphere trajectory for selected (ε, f).
    Shows evolution of |0⟩ under the pulse.
    """
    t_arr, phi_arr = _get_pulse_params(sequence)
    traj = _evolve_trajectory(t_arr, phi_arr, epsilon, f)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere (reduced resolution for faster rendering)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 12)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color="gray")

    # Trajectory
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "b-", linewidth=2, label="Trajectory")
    ax.scatter([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], c="purple", s=80, label=r"$|0\rangle$")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], c="gold", s=80, label="Final")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(rf"Bloch sphere — $\epsilon={epsilon}$, $f={f}$ ({sequence})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig2_fidelity_at_point(epsilon: float, f: float, sequence: str = "rCinBB") -> float:
    """Fidelity at a single (ε, f) point. For display in UI."""
    t_arr, phi_arr = _get_pulse_params(sequence)
    U_ideal = _U(THETA_TARGET, PHI_TARGET)
    seq = _V(0, 0, 0, 0)
    for n in range(len(t_arr)):
        seq = _V(t_arr[n], phi_arr[n], f, epsilon) @ seq
    return _avg_gate_fidelity(seq, U_ideal)


def fig2_pulse_diagram(sequence: str = "rCinBB") -> plt.Figure:
    """
    Interactive Fig. 2(a): Control pulse sequence (I/Q or amplitude/phase vs time).
    Shows rCinBB (6 segments) or SAFE-GRAPE (from cache) pulse shape.
    """
    if sequence == "rCinBB":
        # rCinBB: 6 segments with constant amplitude per segment
        k = np.arcsin(np.sin(THETA_TARGET / 2) / 2)
        theta_ig = [
            np.pi, 2 * np.pi, np.pi,
            2 * np.pi + THETA_TARGET / 2 - k, 2 * np.pi - 2 * k, THETA_TARGET / 2 - k,
        ]
        phi_ig = [
            PHI_TARGET + np.arccos(-THETA_TARGET / 4 / np.pi),
            PHI_TARGET + 3 * np.arccos(-THETA_TARGET / 4 / np.pi),
            PHI_TARGET + np.arccos(-THETA_TARGET / 4 / np.pi),
            PHI_TARGET, PHI_TARGET + np.pi, PHI_TARGET,
        ]
        # Build step plot: t_edges and I,Q per segment
        t_edges = np.concatenate([[0], np.cumsum(np.array(theta_ig) / OMEGA)])
        I_vals = np.array([np.cos(p) for p in phi_ig])
        Q_vals = np.array([np.sin(p) for p in phi_ig])
        t_plot = np.repeat(t_edges, 2)[1:-1] * 1e9  # ns, step edges
        I_plot = np.repeat(I_vals, 2)
        Q_plot = np.repeat(Q_vals, 2)
    elif sequence == "SAFE-GRAPE":
        if not _safe_grape_available():
            raise FileNotFoundError(
                "SAFE-GRAPE cache not found. Run: python scripts/generate_safe_grape_cache.py"
            )
        data = np.load(_SAFE_GRAPE_CACHE)
        I_plot = np.asarray(data["I"])
        Q_plot = np.asarray(data["Q"])
        delta_t = float(data["delta_t"])
        t_plot = np.arange(len(I_plot)) * delta_t * 1e9  # ns
    else:
        raise ValueError(f"Unknown sequence: {sequence}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, gridspec_kw={"hspace": 0.15})

    # I/Q
    axs[0].plot(t_plot, I_plot, color="C0", label="I")
    axs[0].plot(t_plot, Q_plot, color="C1", linestyle="--", label="Q")
    axs[0].set_ylabel("I, Q")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    # Amplitude
    amp = np.sqrt(I_plot**2 + Q_plot**2)
    axs[1].plot(t_plot, amp, color="C2")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(alpha=0.3)

    # Phase
    phi_plot = np.arctan2(Q_plot, I_plot)
    axs[2].plot(t_plot, phi_plot, color="C3")
    axs[2].set_ylabel("Phase [rad]")
    axs[2].set_xlabel("Time [ns]")
    axs[2].grid(alpha=0.3)

    fig.suptitle(f"Control pulse sequence — {sequence} (Ω = {OMEGA/1e6:.0f} Mrad/s)", y=0.98)
    fig.tight_layout()
    return fig
