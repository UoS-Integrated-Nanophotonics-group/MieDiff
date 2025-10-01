# encoding: utf-8
"""
Mie-informed tandem neural network
===================================

Here, we demonstrate how to train a design generator network
capable to suggest core-shell particles with specific spectral response
using PyMieDiff as differentiable forward-evaluator. The training pipeline
follows the "Tandem" model:

target spectrum --> generator NN --> design --> Mie --> real spectrum

training loss is: MSE(target spec., real spec.)


author: O. Jackson, P. Wiecha, 06/2025
"""
# %%
# imports
# -------
import time

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn

import pymiediff as pmd


# %%
# setup optimiation target
# ------------------------
# We setup the main configuration here:
# pymiediff backend, torch device, parameter limits and wavelengths

# pymiediff backend to use and torch compute device
backend = "torch"
device = "cpu"

# general config
N_samples = 25000
n_max = 4  # maximum Mie order fixed for performance
eps_env = torch.tensor(1.0, device=device)

lim_r = torch.as_tensor([40, 100], device=device)
lim_n_re = torch.as_tensor([1.5, 4.0], device=device)
lim_n_im = torch.as_tensor([0.0, 0.1], device=device)

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

all_particles = pmd.coreshell.cross_sections(
    k0,
    r_c=r_c,
    eps_c=eps_c,
    r_s=r_s,
    eps_s=eps_s,
    eps_env=eps_env,
    backend=backend,
    n_max=n_max,
)

N_test = 128  # keep a few samples for testing
q_sca_target = all_particles["q_sca"][N_test:].to(dtype=torch.float32)
q_sca_target_test = all_particles["q_sca"][:N_test].to(dtype=torch.float32)

plt.plot(q_sca_target[30].detach().cpu().numpy())  # plot some test sample


# %%
# Neural network classes / functions
# ----------------------------------
# define the network model (simple MLP) and training loop
class FullyConnected(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.fc_in = nn.Linear(len(k0), hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu1(x)
        x = self.fc_1(x)
        x = self.relu2(x)
        x = self.fc_2(x)
        x = self.relu3(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x


def nn_pred_to_mie_geometry(pred):
    # implicit normalization: multiply by user-defined limits
    r_c = lim_r.max() * (pred[:, 0])
    r_s = lim_r.max() * (pred[:, 0] + pred[:, 1])
    n_c = lim_n_re.max() * pred[:, 2] + lim_n_im.max() * (1j * pred[:, 3])
    n_s = lim_n_re.max() * pred[:, 4] + lim_n_im.max() * (1j * pred[:, 5])

    eps_c = torch.ones_like(k0).unsqueeze(0) * n_c.unsqueeze(1) ** 2
    eps_s = torch.ones_like(k0).unsqueeze(0) * n_s.unsqueeze(1) ** 2

    return r_c, r_s, eps_c, eps_s


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    prog_bar = tqdm(enumerate(dataloader), total=size // dataloader.batch_size)
    for i_batch, X in prog_bar:
        # model prediction: generate core-shell particles
        pred = model(X)

        # evaluate Mie
        r_c, r_s, eps_c, eps_s = nn_pred_to_mie_geometry(pred)
        res_mie = pmd.coreshell.cross_sections(
            k0,
            r_c=r_c,
            eps_c=eps_c,
            r_s=r_s,
            eps_s=eps_s,
            eps_env=eps_env,
            backend=backend,
            n_max=n_max,
        )
        q_sca_mie = res_mie["q_sca"].to(dtype=torch.float32)

        # calc. loss
        loss = loss_fn(q_sca_mie, X)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if i_batch % 100 == 0:
        loss, current = loss.item(), i_batch * dataloader.batch_size + len(X)
        prog_bar.set_description(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %%
# training the Mie-informed network
# ---------------------------------
# here we use some simple, manually optimized training schedule.

model = FullyConnected().to(device)

confs = [
    dict(bs=32, lr=1e-4, n_ep=5),
    dict(bs=64, lr=1e-4, n_ep=5),
    dict(bs=128, lr=1e-4, n_ep=6),
    dict(bs=256, lr=1e-5, n_ep=6),
]

t_start = time.time()
for conf in confs:
    learning_rate = conf["lr"]
    batch_size = conf["bs"]
    epochs = conf["n_ep"]
    print("-------------------------------")
    print(f"LR={learning_rate}, batch_size={batch_size}")
    print("-------------------------------")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = torch.utils.data.DataLoader(q_sca_target, batch_size=batch_size)
    for t in range(epochs):
        print(f"Epoch {t+1}, time={time.time()-t_start:.2f}s")
        train_loop(train_dataloader, model, loss_fn, optimizer)
print("Done!")


# %%
# test the network
# ----------------
# Do some qualitative tests:
# Let the trained network predict some particle geometries and compare
# their Mie spectra with the traget spectrum.

# pick a few of the training samples for testing.
# Note: Ideally tests should be done on separate samples!
sca_test = q_sca_target_test
pred = model(sca_test)

# evaluate Mie
r_c_test, r_s_test, eps_c_test, eps_s_test = nn_pred_to_mie_geometry(pred)
res_mie = pmd.coreshell.cross_sections(
    k0,
    r_c=r_c_test,
    eps_c=eps_c_test,
    r_s=r_s_test,
    eps_s=eps_s_test,
    eps_env=eps_env,
    n_max=n_max,
)

# plot
i_plot = np.random.randint(len(sca_test), size=4)
plt.figure(figsize=(12, 10))
for i_n, i in enumerate(i_plot):
    plt.subplot(2, 2, i_n + 1)
    plt.plot(
        wl0.detach().cpu().numpy(),
        sca_test[i].detach().cpu().numpy(),
        label="reference",
    )
    plt.plot(
        wl0.detach().cpu().numpy(),
        res_mie["q_sca"][i].detach().cpu().numpy(),
        label="predicted particle",
    )
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("scat. efficiency")
plt.show()

# sphinx_gallery_thumbnail_number = 2
