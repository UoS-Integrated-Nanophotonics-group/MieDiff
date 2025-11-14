# encoding: utf-8
"""
PyMieDiff + TorchGDM: Autodiff Multi-scattering
================================================

Multi‑particle scattering demo using PyMieDiff and TorchGDM.

This script demonstrates how to:
  • Define a coated sphere (core + shell) with PyMieDiff.
  • Convert the particle to TorchGDM structures (full‑GPM and dipole‑only).
  • Run single‑particle extinction simulations and compare to Mie theory.
  • Visualise near‑field intensities for Mie, full‑GPM, and dipole models.
  • Set up a small cluster of identical particles, simulate with TorchGDM,
    and benchmark against a T‑matrix solution from the *treams* package.

Dependencies
------------
- torch, numpy, matplotlib
- torchgdm
- pymiediff
- treams (optional, used for the T‑matrix reference)

Notes
------------
* TorchGDM paper: 
  S. Ponomareva et al. SciPost Physics Codebases 60 (2025)
  doi: 10.21468/SciPostPhysCodeb.60
* torchgdm: https://gitlab.com/wiechapeter/torchgdm
* treams: https://github.com/tfp-photonics/treams

author: P. Wiecha, 11/2025
"""
# %%
# imports
# -------

# %%
# test setup
# ----------
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torchgdm as tg
import pymiediff as pmd

# %%
# Mie particle config
# -------------------
wl0 = torch.as_tensor([600, 700])
k0 = 2 * torch.pi / wl0

r_core = 150.0
r_shell = r_core + 100.0
n_core = 4.0
n_shell = 3.0
n_env = 1.0

mat_core = pmd.materials.MatConstant(n_core**2)
mat_shell = pmd.materials.MatConstant(n_shell**2)

# - setup the particle
particle = pmd.Particle(
    r_core=r_core,
    r_shell=r_shell,
    mat_core=mat_core,
    mat_shell=mat_shell,
    mat_env=n_env,
)
print(particle)


# convert to torchGDM structures
# dipole order only: fast, but inaccurate for large particles
structMieGPM_dp = pmd.helper.tg.StructAutodiffMieEffPola3D(particle, wavelengths=wl0)

# full multipole order: accurate, but slow model setup
structMieGPM_gpm = pmd.helper.tg.StructAutodiffMieGPM3D(
    particle, r_gpm=36, wavelengths=wl0
)


# %%
# torchGDM simulation
# -------------------

# - illumination: normal incidence linear pol. plane wave
e_inc_list = [tg.env.freespace_3d.PlaneWave(e0p=1, e0s=0, inc_angle=0)]
env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=n_env**2)

sim_single = tg.simulation.Simulation(
    structures=[structMieGPM_gpm],
    environment=env,
    illumination_fields=e_inc_list,
    wavelengths=wl0,
)
sim_single.run()

# extinction cross section comparison
cs_mie = particle.get_cross_sections(k0=k0)
cs = sim_single.get_spectra_crosssections()

print("")
print("Evaluated wavelengths (nm):")
print(f"     {wl0}")
print("PyMieDiff extinction cross section (nm^2):")
print(f"     {cs_mie["cs_ext"]}")
print("TorchGDM extinction cross section (nm^2):")
print(f"     {cs["ecs"][:,0]}")
print(f"rel. errors:")
print(f"     {[cs_mie["cs_ext"][i] / cs["ecs"][i] for i in range(len(wl0))]}")

# %%
# nearfield comparison
# --------------------
# single sphere :
# compare Mie theory and result for the torchGDM models
#
# Note: the torchGDM models don't support fields inside the particles,
# therefore we calculate fields in a plane next to the particles.

# also compare to dipole-approx. model
sim_single_dp = tg.simulation.Simulation(
    structures=[structMieGPM_dp],
    environment=env,
    illumination_fields=e_inc_list,
    wavelengths=wl0,
)
sim_single_dp.run()



# field calculation grid
projection = "xz"
r_probe = tg.tools.geometry.coordinate_map_2d_square(
    1000, 51, r3=500, projection=projection
)["r_probe"]

# which wavelength to plot
idx_wl = 0

# get nearfields
nf_mie = particle.get_nearfields(k0=k0[idx_wl], r_probe=r_probe)
nf_sim = sim_single.get_nearfield(wl0[idx_wl], r_probe=r_probe)
nf_sim_dp = sim_single_dp.get_nearfield(wl0[idx_wl], r_probe=r_probe)


# --- plot
def plot_circles(radii, pos=None, im=None):
    if pos is None:
        pos = [[0, 0]] * len(radii)

    for r, p in zip(radii, pos):
        rect = Circle(
            xy=p,
            radius=r,
            linewidth=1,
            edgecolor="k",
            facecolor="none",
            alpha=0.5,
        )
        plt.gca().add_patch(rect)
    if im is not None:
        cb = plt.colorbar(im, label="")


plt.figure(figsize=(10, 2.5))

# - Mie
plt.subplot(131, title="Mie - $|E_s|^2 / |E_0|^2$")
im = tg.visu2d.scalarfield(
    (torch.abs(nf_mie["E_s"]) ** 2).sum(-1),
    positions=r_probe,
    projection=projection,
)
clim = im.get_clim()
plot_circles([r_core, r_shell], im=im)

# - torchGDM - GPM
plt.subplot(132, title="torchGDM - GPM")
im = tg.visu2d.scalarfield(
    nf_sim["sca"].get_efield_intensity()[0],
    positions=r_probe,
    projection=projection,
)
im.set_clim(clim)
plot_circles([r_core, r_shell], im=im)

# - torchGDM - dipole order
plt.subplot(133, title="dipole order only")
im = tg.visu2d.scalarfield(
    nf_sim_dp["sca"].get_efield_intensity()[0],
    positions=r_probe,
    projection=projection,
)
plot_circles([r_core, r_shell], im=im)


plt.tight_layout()
plt.show()


# %%
# multiple particles
# ------------------
# compare Mie theory and T-Matrix solver (treams)

# create simple multi-particle configuration: 5 particles on a circle
pos_multi = tg.tools.geometry.coordinate_map_1d_circular(r=700, n_phi=5)["r_probe"]

sim_multi = tg.simulation.Simulation(
    structures=[structMieGPM_gpm.copy(pos_multi)],
    environment=env,
    illumination_fields=e_inc_list,
    wavelengths=wl0,
)
sim_multi.run()

nf_sim_multi = sim_multi.get_nearfield(wl0[idx_wl], r_probe=r_probe)

# %%
# T-matrix multi-particle
# -----------------------
# now we plot the nearfields in a plane parallel to the particle cluster
# and compare to the fields calculated with the treams T-matrix package.
#
# Note: treams and GPM models do not support fields inside the particles,
# so we calculate fields outside of the particles.

import treams

t0 = time.time()

k0_treams = k0[idx_wl].cpu().numpy()

materials = [
    treams.Material(n_core**2),
    treams.Material(n_shell**2),
    treams.Material(n_env**2),
]
lmax = 5
spheres = [
    treams.TMatrix.sphere(lmax, k0_treams, [r_core, r_shell], materials)
    for i in range(len(pos_multi))
]
tm = treams.TMatrix.cluster(spheres, pos_multi.cpu().numpy()).interaction.solve()

e0_pol = [1, 0, 0]
inc = treams.plane_wave([0, 0, k0_treams], pol=e0_pol, k0=tm.k0, material=tm.material)
sca = tm @ inc.expand(tm.basis)

# calc nearfield
grid = r_probe.cpu().numpy()
e_sca_treams = sca.efield(grid)
e_inc_treams = inc.efield(grid)
e_tot_treams = e_sca_treams + e_inc_treams
intensity = np.sum(np.abs(e_sca_treams) ** 2, -1)

print("total time treams: {:.2f}s".format(time.time() - t0))

# %%
# plot the multi-particle nearfields
# ----------------------------------

plt.figure(figsize=(7, 2.5))

plt.subplot(121, title="PyMieDiff + torchGDM")
im = tg.visu2d.scalarfield(
    nf_sim_multi["sca"].get_efield_intensity()[0],
    positions=r_probe,
    projection=projection,
)
clim = im.get_clim()
plot_circles(
    len(pos_multi) * [r_core, r_shell],
    pos_multi[:, [0, 2]].repeat_interleave(2, dim=0),
    im=im,
)

plt.subplot(122, title="treams (T-Matrix)")
im = tg.visu2d.scalarfield(
    intensity,
    positions=r_probe,
    projection=projection,
)
im.set_clim(clim)
plot_circles(
    len(pos_multi) * [r_core, r_shell],
    pos_multi[:, [0, 2]].repeat_interleave(2, dim=0),
    im=im,
)

plt.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 2
