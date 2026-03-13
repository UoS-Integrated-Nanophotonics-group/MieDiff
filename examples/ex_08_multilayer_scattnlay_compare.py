# encoding: utf-8
"""
multilayer spectrum vs scattnlay
================================

Simple 4-layer scattering spectrum comparison between
- pymiediff (backend="pena")
- scattnlay

author: P. Wiecha, 03/2026
"""

# %%
# imports
# -------
import numpy as np
import matplotlib.pyplot as plt
import torch

import pymiediff as pmd

try:
    from scattnlay import scattnlay
except ImportError as exc:
    raise ImportError(
        "This example requires `scattnlay`. Install via `pip install scattnlay`."
    ) from exc


# %%
# setup 4-layer sphere
# --------------------
wl0 = torch.linspace(1095.0, 1100.0, 180, dtype=torch.float64)  # nm
k0 = 2 * torch.pi / wl0

n_env = 1.0

# outer radii of each layer (nm), from core to shell
r_layers = torch.tensor([135.0, 365.0, 395.0, 5630.0], dtype=torch.float64)

# refractive indices of the layers (constant over wavelength for this demo)
n_layers = torch.tensor(
    [2.1 + 0.15j, 1.75 + 0.00j, 0.45 + 5.06j, 3.62 + 0.0j],
    dtype=torch.complex128,
)
eps_layers = n_layers**2

# truncation order (None for auto)
n_max = None

# get smallest size parameter of the configs
x_l = 2 * torch.pi * n_layers[-1].abs() * r_layers[-1] / wl0.max()
print(f"smallest size parameter of the simulations: {x_l:.2f}")

# %%
# pymiediff (multilayer, pena backend)
# ------------------------------------
cs_pmd = pmd.multishell.cross_sections(
    k0=k0,
    r_layers=r_layers,
    eps_layers=eps_layers,
    eps_env=n_env**2,
    backend="pena",
    n_max=n_max,
)
# resolved internal truncation order (Wiscombe estimate if n_max is None)
n_max_use = int(torch.as_tensor(cs_pmd["n_max"]).item())
print("n_max: ", n_max_use)

q_sca_pmd = cs_pmd["q_sca"].squeeze().detach().cpu().numpy()
q_ext_pmd = cs_pmd["q_ext"].squeeze().detach().cpu().numpy()


# %%
# scattnlay reference
# -------------------
wl_np = wl0.detach().cpu().numpy()
k0_np = 2.0 * np.pi / wl_np
r_np = r_layers.detach().cpu().numpy()
n_np = n_layers.detach().cpu().numpy()

q_sca_scnl = np.zeros_like(wl_np, dtype=np.float64)
q_ext_scnl = np.zeros_like(wl_np, dtype=np.float64)

for i_wl, _k0 in enumerate(k0_np):
    x_layers = _k0 * n_env * r_np
    m_layers = n_np / n_env
    _, qext, qsca, *_ = scattnlay(x_layers, m_layers, nmax=n_max_use)
    q_sca_scnl[i_wl] = np.real(qsca)
    q_ext_scnl[i_wl] = np.real(qext)


# %%
# plot comparison
# ---------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

ax[0].plot(wl_np, q_sca_pmd, label="pymiediff (pena)", lw=2)
ax[0].plot(wl_np, q_sca_scnl, "--", label="scattnlay", lw=1.5)
ax[0].set_title("Scattering Efficiency")
ax[0].set_xlabel("wavelength (nm)")
ax[0].set_ylabel(r"$Q_{sca}$")
ax[0].legend()

ax[1].plot(wl_np, q_ext_pmd, label="pymiediff (pena)", lw=2)
ax[1].plot(wl_np, q_ext_scnl, "--", label="scattnlay", lw=1.5)
ax[1].set_title("Extinction Efficiency")
ax[1].set_xlabel("wavelength (nm)")
ax[1].set_ylabel(r"$Q_{ext}$")
ax[1].legend()

plt.show()


# %%
# angular radiation pattern at fixed wavelength
# ---------------------------------------------
wl_ang = torch.tensor([1000.0], dtype=torch.float64)  # nm
k0_ang = 2 * torch.pi / wl_ang
theta = torch.linspace(0.0, torch.pi, 360, dtype=torch.float64)

# - pymiediff
ang_pmd = pmd.multishell.angular_scattering(
    k0=k0_ang,
    theta=theta,
    r_layers=r_layers,
    eps_layers=eps_layers,
    eps_env=n_env**2,
    backend="pena",
    n_max=n_max_use,
)

# remove singleton particle / wavelength dims for plotting
i_unpol_pmd = ang_pmd["i_unpol"].squeeze().detach().cpu().numpy()
i_par_pmd = ang_pmd["i_par"].squeeze().detach().cpu().numpy()
i_per_pmd = ang_pmd["i_per"].squeeze().detach().cpu().numpy()

# - scattnlay (same particle, same wavelength, same angular grid)
theta_np = theta.detach().cpu().numpy()
x_layers_ang = (k0_ang.item() * n_env) * r_np
m_layers_ang = n_np / n_env
_, _, _, _, _, _, _, _, s1_scnl, s2_scnl = scattnlay(
    x_layers_ang, m_layers_ang, theta=theta_np, nmax=n_max_use
)
i_per_scnl = np.abs(s1_scnl) ** 2
i_par_scnl = np.abs(s2_scnl) ** 2
i_unpol_scnl = 0.5 * (i_per_scnl + i_par_scnl)

# - cartesian comparison plot
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
theta_deg = theta_np * 180.0 / np.pi

ax[0].plot(theta_deg, i_unpol_pmd, label="pymiediff", lw=2)
ax[0].plot(theta_deg, i_unpol_scnl, "--", label="scattnlay", lw=1.5)
ax[0].set_title(r"$i_{unpol}$")
ax[0].set_xlabel("theta (deg)")
ax[0].set_ylabel("intensity (a.u.)")
ax[0].legend()

ax[1].plot(theta_deg, i_par_pmd, label="pymiediff", lw=2)
ax[1].plot(theta_deg, i_par_scnl, "--", label="scattnlay", lw=1.5)
ax[1].set_title(r"$i_{par}$")
ax[1].set_xlabel("theta (deg)")

ax[2].plot(theta_deg, i_per_pmd, label="pymiediff", lw=2)
ax[2].plot(theta_deg, i_per_scnl, "--", label="scattnlay", lw=1.5)
ax[2].set_title(r"$i_{per}$")
ax[2].set_xlabel("theta (deg)")

fig.suptitle(f"Angular pattern comparison at {wl_ang.item():.0f} nm")
plt.show()
