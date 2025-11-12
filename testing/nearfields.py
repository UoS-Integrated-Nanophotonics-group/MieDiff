"""
Fully-differentiable scattered near-field
"""

# %%
import time
import torch


import matplotlib.pyplot as plt
import torch
import pymiediff as pmd

# %%
# setup
# -----
# we setup the particle dimension and materials as well as the environemnt.
# This is then wrapped up in an instance of `Particle`.

# - config
# wl0 = torch.linspace(500, 1000, 5)
wl0 = torch.as_tensor([600])
k0 = 2 * torch.pi / wl0

r_core = 150
r_shell = r_core + 150
n_core = 1.5
n_shell = 2.55
n_env = 1.5

# wl0 = torch.as_tensor([650])
# r_core = 250
# r_shell = r_core + 150
# n_core = 3.5
# n_shell = 1.50 + 1j
# n_env = 1.0

# r_core = 50
# r_shell = r_core + 80
# n_core = (-9.4277 + 1.5129j) ** 0.5
# n_shell = (15.4524 + 0.1456j) ** 0.5
# n_env = 1.5

mat_core = pmd.materials.MatConstant(n_core**2)
mat_shell = pmd.materials.MatConstant(n_shell**2)

# - setup the particle
p = pmd.Particle(
    r_core=r_core,
    r_shell=r_shell,
    mat_core=mat_core,
    mat_shell=mat_shell,
    mat_env=n_env,
)
print(p)

d_area_plot = 500
x_offset = 0
y_offset = 0
z_offset = 0

x, z = torch.meshgrid(
    torch.linspace(-d_area_plot, d_area_plot, 40),
    torch.linspace(-d_area_plot, d_area_plot, 40),
)
x = x + x_offset
z = z + z_offset
y = torch.ones_like(x) * y_offset
grid = torch.stack([x, y, z], dim=-1)

orig_shape_grid = grid.shape
r_probe = grid.view(-1, 3)

# r_probe = tg.tools.geometry.coordinate_map_2d_square(
#     d_area_plot, n=100, r3=-250, projection="xy"
# )["r_probe"]
# r_probe = r_probe[torch.sum(r_probe**2, -1) > 1.1 * r_shell**2]

k0 = torch.as_tensor(k0, device=p.device)
eps_c, eps_s, eps_env = p.get_material_permittivities(k0)
r_s = p.r_c if (p.r_s is None) else p.r_s
r_c = p.r_c

from pymiediff.special import vsh
from pymiediff.coreshell import mie_coefficients

E_0 = 1
backend = "torch"
precision = "double"
# which_jn = "ratios"
which_jn = "recurrence"
n_max = 10  # None

t0 = time.time()

# --------------------------------
# - evaluate mie coefficients (vectorized)
miecoeff = mie_coefficients(
    k0=k0,
    r_c=r_c,
    eps_c=eps_c,
    r_s=r_s,
    eps_s=eps_s,
    eps_env=eps_env,
    backend=backend,
    precision=precision,
    which_jn=which_jn,
    n_max=n_max,
)

n = miecoeff["n"]
n_max = miecoeff["n_max"]
k = miecoeff["k"]
k0 = miecoeff["k0"]
r_c = miecoeff["r_c"]
r_s = miecoeff["r_s"]
n_sourrounding = miecoeff["n_env"]
a_n = miecoeff["a_n"]
b_n = miecoeff["b_n"]


kc = miecoeff["k0"] * r_c
ks = miecoeff["k0"] * r_s

# - Cartesian positions to spherical coordinates
r, theta, phi = pmd.helper.transform_xyz_to_spherical(
    r_probe[..., 0], r_probe[..., 1], r_probe[..., 2]
)

from pymiediff.special import pi_tau, psi, xi, psi_torch, xi_torch

# - vector spherical harmonics
# M_o1n, M_e1n, N_o1n, N_e1n = vsh(n, k0, n_env, r, theta, phi, kind=3)
theta = theta
kind = 3

# canonicalize n_max
if isinstance(n, torch.Tensor):
    n_max = int(n.max().item())
else:
    n_max = int(n)
assert n_max >= 0

# vectorization:
#   - dim 0: Mie order
#   - dim 1: n particles
#   - dim 2: wavevectors
#   - dim 3: positions
#   - dim 4: field vector components (3)
n_p = r_c.shape[0]
n_k0 = k0.shape[1]
n_pos = theta.shape[0]
full_shape = (n_max, n_p, n_k0, n_pos, 3)

# expand dimensions
# add order, position, vector dim
k = k.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
k0 = k0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
n_sourrounding = n_sourrounding.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

# add position, vector dim
a_n = a_n.unsqueeze(-1).unsqueeze(-1)
b_n = b_n.unsqueeze(-1).unsqueeze(-1)

# add order, particle, wavenumber, vetor dimensions
r = r.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
phi = phi.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
theta = theta.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

# add wavenumber, position, vector dim
r_c = r_c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
r_s = r_s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

# add particle, wavenumber, position, vector dimensions
n = n.view((-1,) + (r.ndim - 1) * (1,))

tvsh = time.time()
M_o1n, M_e1n, N_o1n, N_e1n = pmd.special.vsh(
    n_max, k0, n_sourrounding, r, theta, phi, kind
)
print("total time VSH-pmd: {:.6f}s".format(time.time() - tvsh))


t0 = time.time()
fields_all = pmd.coreshell.nearfields(
    k0=k0,
    r_probe=r_probe,
    r_c=r_core,
    eps_c=eps_c,
    r_s=r_shell,
    eps_s=eps_s,
    eps_env=eps_env,
    n_max=n_max,
)
Es_xyz = fields_all["E_t"]
Hs_xyz = fields_all["H_t"]  # test the H-field
print("total time pmd nearfields: {:.6f}s".format(time.time() - t0))

## %%
i_particle = 0
i_wl = 0
E_sca_mie = Es_xyz[i_particle, i_wl]
E_sca_mie = E_sca_mie.view(orig_shape_grid)
intensity_sca_mie = torch.sum(E_sca_mie.abs() ** 2, dim=-1)
H_sca_mie = Hs_xyz[i_particle, i_wl]
H_sca_mie = H_sca_mie.view(orig_shape_grid)
intensityH_sca_mie = torch.sum(H_sca_mie.abs() ** 2, dim=-1)

# tg.visu2d.scalarfield(
#     torch.sum(Es_xyz[i_particle, i_wl].abs() ** 2, dim=-1), r_probe
# )
# tg.visu2d.scalarfield(Es_xyz[i_particle, i_wl, :, 2].real, r_probe)


# %%

# %% compare to reference
# treams (t-matrix)

import treams
import numpy as np

# eps_spheres = [n_core**2, n_shell**2]
radii = [
    torch.atleast_1d(r_c.squeeze()).numpy()[0],
    torch.atleast_1d(r_s.squeeze()).numpy()[0],
]
# t0 = time.time()

positions = [[0, 0, 0]]

# materials = [treams.Material(_eps) for _eps in eps_spheres] + [
#     treams.Material(torch.atleast_1d(n_env.squeeze()).numpy()[0].real ** 2)
# ]
# lmax = 15
# spheres = [
#     treams.TMatrix.sphere(
#         lmax, torch.atleast_1d(k0.squeeze()).numpy()[i_wl], radii, materials
#     )
# ]
# tm = treams.TMatrix.cluster(spheres, positions).interaction.solve()

# e0_pol = [1, 0, 0]
# inc = treams.plane_wave(
#     [0, 0, torch.atleast_1d(k0.squeeze()).numpy()[i_wl]],
#     pol=e0_pol,
#     k0=tm.k0,
#     material=tm.material,
# )
# sca = tm @ inc.expand(tm.basis)

# calc nearfield
e_sca_treams = np.zeros(E_sca_mie.shape)
# e_sca_treams = sca.efield(grid.numpy())
# # e_sca_treams = sca.hfield(grid.numpy())
# e_sca_treams[(grid**2).sum(-1).numpy()<r_shell**2] = 0.0
# e_tot_treams = inc.efield(grid.numpy()) + sca.efield(grid.numpy())
intensity_sca_treams = np.sum(np.abs(e_sca_treams) ** 2, -1)
# intensity_tot_treams = np.sum(np.abs(e_tot_treams) ** 2, -1)

print("total time treams: {:.2f}s".format(time.time() - t0))

# %% scattnlayer
from scattnlay import scattnlay, fieldnlay

k = k0 * n_env
x = k.squeeze().numpy() * r_core
y = k.squeeze().numpy() * r_shell
m_c = n_core / n_env
m_s = n_shell / n_env
x_list = np.array([x, y])
m_list = np.array([m_c, m_s])
coords = r_probe

t0 = time.time()

terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_list, m_list)
terms, E_scnl, H_scnl = fieldnlay(
    x_list, m_list, *(k.squeeze().numpy() * coords.numpy()).T, nmax=10
)

E_scnl = np.nan_to_num(E_scnl).reshape(tuple(orig_shape_grid))
Z0 = 376.73
H_scnl = n_env* np.nan_to_num(H_scnl).reshape(tuple(orig_shape_grid)) * Z0
# H_scnl[x**2+y**2+z**2 <r_core**2] /= (n_core**2/n_shell**2)
# H_scnl[x**2+y**2+z**2 <r_shell**2] /= n_shell**2


E_sca_mie[np.abs(E_scnl) > 100] = 0.0  # remove singularities
H_sca_mie[np.abs(H_scnl) > 100] = 0.0  # remove singularities
E_sca_mie[np.abs(E_scnl) == 0] = 0.0  # remove singularities
H_sca_mie[np.abs(H_scnl) == 0] = 0.0  # remove singularities
E_scnl[np.abs(E_scnl) > 100] = 0.0  # remove singularities
H_scnl[np.abs(H_scnl) > 100] = 0.0  # remove singularities


intensity_sca_scnl = np.sum(np.abs(E_scnl) ** 2, -1)
intensityH_sca_scnl = np.sum(np.abs(H_scnl) ** 2, -1)
print("total time pmd scattnlay: {:.6f}s".format(time.time() - t0))

# %%
# plot field intensity
# --------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

Ecomp_str = ["x", "y", "z"]


def plot_circles(radii, pos, im=None):
    for r in radii:
        rect = Circle(
            xy=pos[:2],
            radius=r,
            linewidth=1.5,
            edgecolor="k",
            facecolor="C0",
            alpha=0.25,
        )
        plt.gca().add_patch(rect)
    if im is not None:
        cb = plt.colorbar(im, label="")


for i_case in [0, 1]:
    if i_case == 0:
        intensity_mie = intensity_sca_mie
        intensity_scnl = intensity_sca_scnl
        F_mie = E_sca_mie
        F_scnl = E_scnl
    if i_case == 1:
        intensity_mie = intensityH_sca_mie
        intensity_scnl = intensityH_sca_scnl
        F_mie = H_sca_mie
        F_scnl = H_scnl
    fig = plt.figure(figsize=(12, 8.0))

    # ---------------------- pymiediff
    plt.subplot(445, aspect="equal", title="|E|$^2$ pymiediff")
    im = plt.imshow(
        intensity_mie.T,
        extent=(2 * (-d_area_plot, d_area_plot)),
        origin="lower",
    )
    plot_circles(radii, positions[0], im)
    clim = im.get_clim()

    # ---------------------- scattnlay
    plt.subplot(4, 4, 9, aspect="equal", title="|E|$^2$ scattnlay")
    im = plt.imshow(
        intensity_scnl.T,
        extent=(2 * (-d_area_plot, d_area_plot)),
        origin="lower",
    )
    plot_circles(radii, positions[0], im)
    im.set_clim(clim)

    for i in range(3):
        # ---------------------- pymiediff
        plt.subplot(
            4, 4, i + 6, aspect="equal", title=f"Re(E_{Ecomp_str[i]}) pymiediff"
        )
        im = plt.imshow(
            F_mie[..., i].T.real,
            extent=(2 * (-d_area_plot, d_area_plot)),
            origin="lower",
        )
        plot_circles(radii, positions[0], im)
        if max(np.abs(im.get_clim())) < 0.001:
            plt.clim(-0.1, 0.1)
        clim = im.get_clim()

        # ---------------------- scattnlay
        plt.subplot(
            4, 4, i + 10, aspect="equal", title=f"Re(E_{Ecomp_str[i]}) scattnlay"
        )
        im = plt.imshow(
            np.nan_to_num(F_scnl[..., i].T.real),
            extent=(2 * (-d_area_plot, d_area_plot)),
            origin="lower",
        )
        # plt.colorbar()
        plot_circles(radii, positions[0], im)
        # if max(np.abs(im.get_clim())) < 0.001:
        #     plt.clim(-0.1, 0.1)
        im.set_clim(clim)

    plt.tight_layout()
    # plt.savefig('fields_vs_treams.svg')
    plt.show()


for i_case in [0, 1]:
    if i_case == 0:
        intensity_mie = intensity_sca_mie
        intensity_scnl = intensity_sca_scnl
        F_mie = E_sca_mie
        F_scnl = E_scnl
    if i_case == 1:
        intensity_mie = intensityH_sca_mie
        intensity_scnl = intensityH_sca_scnl
        F_mie = H_sca_mie
        F_scnl = H_scnl
    plt.subplot(1, 2, i_case + 1)
    plt.imshow(
        (intensity_scnl - intensity_mie.numpy()).T,
        extent=(2 * (-d_area_plot, d_area_plot)),
        origin="lower",
    )
    plt.colorbar()
plt.show()

# %%
