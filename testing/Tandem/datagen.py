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
import h5py

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

q_sca = all_particles["q_sca"].to(dtype=torch.float32)
# %%
print(q_sca.shape)

plt.plot(q_sca[30].detach().cpu().numpy())


# %%
# save data
# ---------------------

with h5py.File("cs_data.h5", "w") as f:
    f.create_dataset("k0", data=k0.detach().cpu().numpy())
    f.create_dataset("r_c", data=r_c.detach().cpu().numpy())
    # f.create_dataset("d_s", data=d_s)
    f.create_dataset("r_s", data=r_s.detach().cpu().numpy())
    f.create_dataset("n_re", data=n_re.detach().cpu().numpy())
    f.create_dataset("n_im", data=n_im.detach().cpu().numpy())
    f.create_dataset("n", data=n.detach().cpu().numpy())
    f.create_dataset("eps_c", data=eps_c.detach().cpu().numpy())
    f.create_dataset("eps_s", data=eps_s.detach().cpu().numpy())

    f.create_dataset("q_sca", data=q_sca.detach().cpu().numpy())



# %%
