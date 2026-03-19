"""
Generate SAFE-GRAPE pulse cache for the paper explorer.
Run once (requires torch): python scripts/generate_safe_grape_cache.py

Replicates BandwidthAware_SAFEGRAPE.ipynb optimization and saves I, Q to
app/data/safe_grape_pulse.npz for use in Interactive Fig. 2.

Uses reduced params for faster run (~1-2 min). For full fidelity, edit the
constants below (N_TRAINING_SEGMENTS, UPSAMPLING_CAP, etc.).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# Paper-quality params from BandwidthAware_SAFEGRAPE.ipynb
# UPSAMPLING_CAP=15 trades ~4x speed for slightly coarser pulse (still high fidelity)
N_TRAINING_SEGMENTS = 100
BANDWIDTH = 1e8  # Hz
OVERSAMPLING_FACTOR = 50
DELTA_T = 1 / (2 * OVERSAMPLING_FACTOR * BANDWIDTH)
LPF_SIGMA = np.sqrt(np.log(2)) / (2 * np.pi * BANDWIDTH * DELTA_T)
OMEGA = 100e6  # rad/s
OMEGA_MAX = 200e6  # rad/s
THETA = np.pi
PHI = 0.0
PI = np.pi
UPSAMPLING_CAP = 15  # Full=~60, 15 gives ~5 min run with good fidelity
MAX_ITER = 150  # L-BFGS iterations
GRID_SIZE = 11  # Epsilon/f grid (11x11)


def main():
    print("Initializing rCinBB seed and SAFE-GRAPE model...")
    # rCinBB initial guess
    k = np.arcsin(np.sin(THETA / 2) / 2)
    theta_ig = [PI, 2 * PI, PI, 2 * PI + THETA / 2 - k, 2 * PI - 2 * k, THETA / 2 - k]
    phi_ig = [
        PHI + np.arccos(-THETA / 4 / PI),
        PHI + 3 * np.arccos(-THETA / 4 / PI),
        PHI + np.arccos(-THETA / 4 / PI),
        PHI,
        PHI + PI,
        PHI,
    ]
    total_theta = sum(theta_ig)
    total_time = total_theta / OMEGA
    I_ig = torch.zeros(N_TRAINING_SEGMENTS, dtype=torch.float64)
    Q_ig = torch.zeros(N_TRAINING_SEGMENTS, dtype=torch.float64)
    for i in range(6):
        start_idx = int(np.rint(sum(theta_ig[:i]) / total_theta * N_TRAINING_SEGMENTS))
        end_idx = int(np.rint(sum(theta_ig[: i + 1]) / total_theta * N_TRAINING_SEGMENTS))
        I_ig[start_idx:end_idx] = np.cos(phi_ig[i]) * OMEGA / OMEGA_MAX
        Q_ig[start_idx:end_idx] = np.sin(phi_ig[i]) * OMEGA / OMEGA_MAX

    _raw = int(np.rint(total_time / (N_TRAINING_SEGMENTS * DELTA_T)))
    UPSAMPLING_FACTOR = max(1, min(UPSAMPLING_CAP, _raw))
    TIME_STEPS = N_TRAINING_SEGMENTS * UPSAMPLING_FACTOR
    print(f"  TIME_STEPS={TIME_STEPS}, grid={GRID_SIZE}x{GRID_SIZE}")

    # SAFE_GRAPE model (minimal copy from notebook)
    class SAFE_GRAPE(torch.nn.Module):
        def __init__(self, I_ig, Q_ig):
            super().__init__()
            self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
            self.sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
            self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
            self.I = torch.nn.Parameter(I_ig.clone().detach(), requires_grad=True)
            self.Q = torch.nn.Parameter(Q_ig.clone().detach(), requires_grad=True)
            K = int(4 * LPF_SIGMA)
            k_arr = torch.arange(-K, K + 1, dtype=torch.float64)
            h = torch.exp(-0.5 * (k_arr / LPF_SIGMA) ** 2)
            h = h / h.sum()
            self.register_buffer("lpf_kernel", h.view(1, 1, -1))

        def lpf(self, x):
            x_ = x.view(1, 1, -1)
            pad = (self.lpf_kernel.shape[-1] - 1) // 2
            x_pad = torch.nn.functional.pad(x_, (pad, pad), mode="reflect")
            y = torch.nn.functional.conv1d(x_pad, self.lpf_kernel)
            return y.view(-1)

        def get_IQ(self):
            Irep = self.I.repeat_interleave(UPSAMPLING_FACTOR)
            Qrep = self.Q.repeat_interleave(UPSAMPLING_FACTOR)
            If = self.lpf(Irep)
            Qf = self.lpf(Qrep)
            amp = torch.sqrt(If**2 + Qf**2) + 1e-12
            scale = torch.clamp(1 / amp, max=1.0)
            return If * scale, Qf * scale

        def V(self, t, I, Q, f, eps):
            return torch.linalg.matrix_exp(
                -1j
                * OMEGA_MAX
                * t
                / 2
                * ((1 + eps) * (I * self.sigma_x + Q * self.sigma_y) + f * self.sigma_z)
            )

        def U(self, theta, phi):
            return torch.linalg.matrix_exp(
                -1j * theta / 2 * (torch.cos(phi) * self.sigma_x + torch.sin(phi) * self.sigma_y)
            )

        def avg_fidelity(self, U, V):
            return torch.abs(torch.trace(torch.conj(U).T @ V) / 2)

        def forward(self):
            EPSILON_RANGE, F_RANGE = 0.3, 0.3
            EPSILON_N = F_N = GRID_SIZE
            I, Q = self.get_IQ()
            infid = torch.zeros((EPSILON_N, F_N), dtype=torch.float64)
            eps_vals = torch.linspace(-EPSILON_RANGE, EPSILON_RANGE, EPSILON_N)
            f_vals = torch.linspace(-F_RANGE, F_RANGE, F_N)
            for i, eps in enumerate(eps_vals):
                for j, f in enumerate(f_vals):
                    seq = self.V(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), 0.0, 0.0)
                    for n in range(TIME_STEPS):
                        seq = self.V(DELTA_T, I[n], Q[n], f, eps) @ seq
                    infid[i, j] = 1 - self.avg_fidelity(seq, self.U(torch.tensor(THETA), torch.tensor(PHI)))
            return infid.sum()

    grape = SAFE_GRAPE(I_ig, Q_ig)
    optimizer = torch.optim.LBFGS(grape.parameters(), lr=1, max_iter=MAX_ITER)

    _closure_count = [0]  # mutable for closure

    def closure():
        optimizer.zero_grad()
        loss = grape()
        loss.backward()
        _closure_count[0] += 1
        if _closure_count[0] <= 3 or _closure_count[0] % 5 == 0:
            print(f"  L-BFGS eval {_closure_count[0]}: loss = {loss.item():.6f}", flush=True)
        return loss

    print("Running SAFE-GRAPE optimization (~5 min)...")
    optimizer.step(closure)
    print(f"Done. Final loss: {closure().item():.6f}")

    with torch.no_grad():
        If, Qf = grape.get_IQ()
        I = If.cpu().numpy()
        Q = Qf.cpu().numpy()

    out_dir = REPO_ROOT / "app" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "safe_grape_pulse.npz"
    np.savez(out_path, I=I, Q=Q, delta_t=DELTA_T, omega_max=OMEGA_MAX)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
