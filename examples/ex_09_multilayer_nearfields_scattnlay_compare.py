# encoding: utf-8
"""
multilayer near-field vs scattnlay
==================================

Near-field comparison for a 4-layer sphere between
- pymiediff
- scattnlay

author: P. Wiecha, 03/2026
"""

# %%
# imports
# -------
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

import pymiediff as pmd

try:
    from scattnlay import fieldnlay
except ImportError as exc:
    raise ImportError(
        "This example requires `scattnlay`. Install via `pip install scattnlay`."
    ) from exc


# %%
# setup 4-layer sphere and wavelength
# -----------------------------------
wl0 = torch.as_tensor([700.0], dtype=torch.float64)  # nm
k0 = 2 * torch.pi / wl0
n_env = 1.2

# outer radii of each layer (nm)
r_layers = torch.tensor([45.0, 75.0, 85.0, 125.0], dtype=torch.float64)

# refractive indices in each layer
n_layers = torch.tensor(
    [0.45 + 7.55j, 5.5 + 0.1j, 0.45 + 7.55j, 2.1 + 0.0j], dtype=torch.complex128
)
eps_layers = n_layers**2

# resolve truncation using pymiediff internal criterion
mie = pmd.multishell.mie_coefficients(
    k0=k0,
    r_layers=r_layers,
    eps_layers=eps_layers,
    eps_env=n_env**2,
    n_max=None,
)
n_max_use = int(mie["n_max"])
print("Using n_max =", n_max_use)


# %%
# probe grid in xz plane (y=0)
# ----------------------------
d_area_plot = 200.0
x, z = torch.meshgrid(
    torch.linspace(-d_area_plot, d_area_plot, 30),
    torch.linspace(-d_area_plot, d_area_plot, 30),
)
y = torch.zeros_like(x)
grid = torch.stack([x, y, z], dim=-1)
orig_shape = grid.shape
r_probe = grid.view(-1, 3)


# %%
# pymiediff near-field
# --------------------
t0 = time.time()

res_pmd = pmd.multishell.nearfields(
    k0=k0,
    r_probe=r_probe,
    r_layers=r_layers,
    eps_layers=eps_layers,
    eps_env=n_env**2,
    n_max=n_max_use,
)
E_pmd = res_pmd["E_t"][0, 0].reshape(orig_shape).detach().cpu().numpy()
I_pmd = np.sum(np.abs(E_pmd) ** 2, axis=-1)

t1 = time.time()
print(f"PyMieDiff field time: {1000*(t1-t0):.1f}ms")


# %%
# scattnlay near-field
# --------------------
k = float((k0 * n_env).item())
x_list = (k * r_layers.detach().cpu().numpy()).astype(np.float64)
m_list = (n_layers.detach().cpu().numpy() / n_env).astype(np.complex128)

t0 = time.time()
_, E_scnl, _ = fieldnlay(
    x_list,
    m_list,
    *(k * r_probe.detach().cpu().numpy()).T,
    nmax=n_max_use,
)
E_scnl = np.nan_to_num(E_scnl).reshape(orig_shape)
I_scnl = np.sum(np.abs(E_scnl) ** 2, axis=-1)

t1 = time.time()
print(f"Scattnlay field time: {1000*(t1-t0):.1f}ms")

# %%
# comparison plots
# ----------------
eps = 1e-12
rel_diff = np.abs(I_pmd - I_scnl) / np.maximum(I_scnl, eps)


def _plot_layers(ax):
    for rr in r_layers.detach().cpu().numpy():
        circ = Circle((0.0, 0.0), rr, ec="w", fc="none", lw=0.8, alpha=0.9)
        ax.add_patch(circ)


fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

im0 = ax[0].imshow(
    I_pmd.T,
    origin="lower",
    extent=(-d_area_plot, d_area_plot, -d_area_plot, d_area_plot),
)
ax[0].set_title("pymiediff |E|$^2$")
ax[0].set_xlabel("x (nm)")
ax[0].set_ylabel("z (nm)")
_plot_layers(ax[0])
plt.colorbar(im0, ax=ax[0], fraction=0.046)

im1 = ax[1].imshow(
    I_scnl.T,
    origin="lower",
    extent=(-d_area_plot, d_area_plot, -d_area_plot, d_area_plot),
)
ax[1].set_title("scattnlay |E|$^2$")
ax[1].set_xlabel("x (nm)")
ax[1].set_ylabel("z (nm)")
_plot_layers(ax[1])
plt.colorbar(im1, ax=ax[1], fraction=0.046)

im2 = ax[2].imshow(
    np.clip(rel_diff.T, 0.0, 1.0),
    origin="lower",
    extent=(-d_area_plot, d_area_plot, -d_area_plot, d_area_plot),
    cmap="magma",
)
ax[2].set_title("relative diff. |E|$^2$")
ax[2].set_xlabel("x (nm)")
ax[2].set_ylabel("z (nm)")
_plot_layers(ax[2])
plt.colorbar(im2, ax=ax[2], fraction=0.046, label="|Δ| / max(ref, eps)")

plt.show()
