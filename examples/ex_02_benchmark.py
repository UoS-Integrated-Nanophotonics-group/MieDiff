# encoding: utf-8
"""
benchmark
=========

Direct Mie evluation benchmark against other tools

author: P. Wiecha, 03/2025
"""
# %%
# imports
# -------
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import pymiediff as pmd

import pymiecs as pmc
import treams


# %%
# setup test case
# ---------------
# we setup a simple test case

# - config
N_wl = 1000
wl0 = torch.linspace(500, 1500, N_wl)
k0 = 2 * torch.pi / wl0

r_core = 60.0
r_shell = 170.0
n_core = 2.5
n_shell = 4.5
mat_core = pmd.materials.MatConstant(n_core**2)
mat_shell = pmd.materials.MatConstant(n_shell**2)
n_env = 1.0


# %%
# Mie from `treams`
# -----------------
# we benchmark against the t-matrix toolkit `treams`:
# https://github.com/tfp-photonics/treams
# write a simple wrapper for the `treams` Mie code.

def mie_ab_sphere_treams(k0: np.ndarray, radii: list, materials: list, n_max=10):
    """Mie scattering via package `treams`, for vacuum environment"""
    assert len(radii) == len(materials)

    # embedding medium: vacuum
    eps_env = 1.0

    # main Mie extraction and setup
    a_n = np.zeros((len(k0), n_max), dtype=np.complex128)
    b_n = np.zeros_like(a_n)
    for i_wl, _k0 in enumerate(k0):

        # core and shell materials
        mat_treams = []
        for eps_mat in materials:
            mat_treams.append(treams.Material(eps_mat))

        # treams convention: environment material last
        mat_treams.append(treams.Material(eps_env))

        for n in range(1, n_max + 1):
            miecoef = treams.coeffs.mie(n, _k0 * np.array(radii), *zip(*mat_treams))
            a_n[i_wl, n - 1] = -miecoef[0, 0] - miecoef[0, 1]
            b_n[i_wl, n - 1] = -miecoef[0, 0] + miecoef[0, 1]

    # scattering
    Qs_mie = np.zeros(len(k0))
    for n in range(n_max):
        Qs_mie += (2 * (n + 1) + 1) * (np.abs(a_n[:, n]) ** 2 + np.abs(b_n[:, n]) ** 2)
    Qs_mie *= 2 / (k0 * max(radii)).real ** 2

    return dict(
        a_n=a_n,
        b_n=b_n,
        q_sca=Qs_mie,
        wavelength=2 * np.pi / k0,
    )


t0 = time.time()
cs_treams = mie_ab_sphere_treams(
    k0=k0.numpy(), radii=[r_core, r_shell], materials=[n_core**2, n_shell**2], n_max=10
)
t_treams = time.time() - t0

# %%
# PyMieCS
# -------
# https://gitlab.com/wiechapeter/pymiecs

# - calculate efficiencies
t0 = time.time()
cs_pmc = pmc.Q(
    k0, r_core=r_core, n_core=n_core, r_shell=r_shell, n_shell=n_shell, n_max=10
)
t_pymiecs = time.time() - t0

# %%
# PyMieDiff
# ---------
# Differentiable Mie

# - setup the particle
p = pmd.Particle(
    r_core=r_core,
    r_shell=r_shell,
    mat_core=mat_core,
    mat_shell=mat_shell,
    mat_env=n_env,
)

t0 = time.time()
cs_pmd = p.get_cross_sections(k0, n_max=10)
t_pymiediff_scipy = time.time() - t0

# using native torch Mie coefficients
t0 = time.time()
cs_pmd_torch = p.get_cross_sections(k0, n_max=10, backend="torch")
t_pymiediff_torch = time.time() - t0


# %%
# results
# -------

# - plot
plt.figure()
plt.plot(cs_pmd["wavelength"], cs_pmd["q_sca"], label="PyMieDiff")
plt.plot(cs_pmd["wavelength"], cs_pmd_torch["q_sca"], label="PyMieDiff-torch", dashes=[1,1])
plt.plot(cs_treams["wavelength"], cs_treams["q_sca"], label="treams", dashes=[2, 2])
plt.plot(cs_pmd["wavelength"], cs_pmc["qsca"], label="pymiecs", dashes=[1, 2])
plt.xlabel("wavelength (nm)")
plt.ylabel("$Q_{sca}$")
plt.legend()
plt.tight_layout()
# plt.savefig("ex_02.svg", dpi=300)
plt.show()


print("calculated {} wavelengths.".format(N_wl))
print(
    "time PyMieDiff (scipy backend): {:.4f} ms / wl".format(
        (t_pymiediff_scipy) * 1e3 / N_wl
    )
)
print(
    "time PyMieDiff (torch backend): {:.4f} ms / wl".format(
        (t_pymiediff_torch) * 1e3 / N_wl
    )
)
print("time pymiecs:                   {:.4f} ms / wl".format((t_pymiecs) * 1e3 / N_wl))
print("time treams:                    {:.4f} ms / wl".format((t_treams) * 1e3 / N_wl))
