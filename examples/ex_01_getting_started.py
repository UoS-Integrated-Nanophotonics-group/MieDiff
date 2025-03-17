# encoding: utf-8
"""
getting started
===============

demonstration of basic pymiediff functionality: 

- Extinction, scattering and absorption spectra
- Angular scattering patterns

author: P. Wiecha, 03/2025
"""
# %%
# imports
# -------

import matplotlib.pyplot as plt
import torch
import pymiediff as pmd
import os

# %%
# setup
# -----
# we setup the particle dimension and materials as well as the environemnt.
# This is then wrapped up in an instance of `Particle`.

# - config
wl0 = torch.linspace(500, 1000, 50)
k0 = 2 * torch.pi / wl0

r_core = 70.0
r_shell = 100.0
mat_core = pmd.materials.MatDatabase("Si")
mat_shell = pmd.materials.MatDatabase("Ge")
n_env = 1.0

# - setup the particle
p = pmd.Particle(
    r_core=r_core,
    r_shell=r_shell,
    mat_core=mat_core,
    mat_shell=mat_shell,
    mat_env=n_env,
)
print(p)

# %%
# efficiency spectra
# ------------------
# Calculate extinction, scattering and absorption spectra

cs = p.get_cross_sections(k0)

# - plot
plt.figure()
plt.plot(cs["wavelength"], cs["q_ext"], label="$Q_{ext}$")
plt.plot(cs["wavelength"], cs["q_sca"], label="$Q_{sca}$")
plt.plot(cs["wavelength"], cs["q_abs"], label="$Q_{abs}$")
plt.xlabel("wavelength (nm)")
plt.ylabel("Efficiency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),"ex_01_efficiency.png"))
plt.show()

# %%
# angular scattering
# ------------------
# Calculate angular radiation patterns of the scattering

theta = torch.linspace(0.0, 2 * torch.pi, 100)
angular = p.get_angular_scattering(k0, theta)

# - plot every 5th wavelength
plt.figure(figsize=(12, 2))
for i, i_k0 in enumerate(range(len(k0))[::5]):
    ax = plt.subplot(1, 10, i + 1, polar=True)
    plt.title(f"{wl0[i_k0]:.1f} nm")
    ax.plot(angular["theta"], angular["i_unpol"][i_k0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),"ex_01_angular.png"))
plt.show()
