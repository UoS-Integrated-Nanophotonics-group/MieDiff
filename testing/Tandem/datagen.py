# %%
# imports
# -------

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

# %%
device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else
print(device)

# %%
# setup optimiation target
# ------------------------
# We setup the main configuration here:
# pymiediff backend, torch device, parameter limits and wavelengths

# pymiediff backend to use and torch compute device
backend = "torch"
device = "cpu"

# general config
N_samples = 20000
n_max = 3  # maximum Mie order fixed for performance
eps_env = torch.tensor(1.0, device=device)

lim_r = torch.as_tensor([40, 100], device=device)
lim_n_re = torch.as_tensor([1.5, 3.5], device=device)
lim_n_im = torch.as_tensor([0.0, 0.02], device=device)

wl0 = torch.linspace(400, 800, 40, device=device)
k0 = 2 * torch.pi / wl0

# %%
# generate reference spectra
# --------------------------
# we generate a large number of reference Mie spectra for existing
# particles, that will be used as design targets during training.
#
# Note: this step could also be done without any physics knowledge,
# for example with artificial spectra (e.g. Lorentzians), or a
# scattering maximization loss.

torch.manual_seed(42)

# datagen: generate existing spectra (won't use the geometries for training)
r_c = torch.rand((N_samples), device=device) * torch.diff(lim_r)[0] + lim_r[0]
d_s = torch.rand((N_samples), device=device) * torch.diff(lim_r)[0] + lim_r[0]
r_s = r_c + d_s
n_re = torch.rand((N_samples, 2), device=device) * torch.diff(lim_n_re)[0] + lim_n_re[0]
n_im = torch.rand((N_samples, 2), device=device) * torch.diff(lim_n_im)[0] + lim_n_im[0]
n = n_re + 1j * n_im

# low-level API: permittivity required as spectra (for vectorization)
eps_c = torch.ones_like(k0).unsqueeze(0) * n[:, 0].unsqueeze(1) ** 2
eps_s = torch.ones_like(k0).unsqueeze(0) * n[:, 1].unsqueeze(1) ** 2

all_particles = pmd.farfield.cross_sections(
    k0,
    r_c=r_c,
    eps_c=eps_c,
    r_s=r_s,
    eps_s=eps_s,
    eps_env=eps_env,
    backend=backend,
    n_max=n_max,
)

q_sca_target = all_particles["q_sca"].to(dtype=torch.float32)

plt.plot(q_sca_target[30].detach().cpu().numpy())

# %%
# generate data
# -------------

r_c, eps_c, r_s, eps_s = params_to_physical(r_arr, n_arr)

q_sca = []
q_abs = []
q_ext = []


for i in range(sample_num):
    args = (k0.detach().cpu(), r_c[i].detach().cpu(), eps_c[i].detach().cpu(), r_s[i].detach().cpu(), eps_s[i].detach().cpu().cpu())
    result = pmd.farfield.cross_sections(*args, n_max=n_max)
    q_sca.append(result["q_sca"])
    q_abs.append(result["q_abs"])
    q_ext.append(result["q_ext"])

q_sca = torch.stack(q_sca).to(device)
q_abs = torch.stack(q_abs).to(device)
q_ext = torch.stack(q_ext).to(device)


# %%
# define spectra scaling functions
# --------------------------------

lim_q = torch.as_tensor([q_sca.min().item(), q_sca.max().item()], dtype=torch.float, device=device)

print(lim_q)

def spectra_to_normlaised(spectra):
    return (spectra - lim_q.min())/ (lim_q.max() - lim_q.min())

def spectra_to_physical(spectra_n):
    return spectra_n * (lim_q.max() - lim_q.min()) + lim_q.min()


# %%
# make x and y datasets
# ---------------------

x = torch.cat((r_arr, n_arr), dim=0)
y = spectra_to_normlaised(q_sca).T

print("X shape:", x.shape)
print("y shape:", y.shape)

x_meta = torch.cat([lim_r, lim_n_re, lim_n_im])
y_meta = torch.cat([lim_q, wl0])

print("x metadata", x_meta)
print("y metadata", y_meta)

# %%
# save datasets to npy

np.save("x.npy", x)
np.save("y.npy", y)

np.savetxt("x_meta.txt", x_meta)
np.savetxt("y_meta.txt", y_meta)


# %%