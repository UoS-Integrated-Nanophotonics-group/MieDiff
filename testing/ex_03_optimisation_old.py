# encoding: utf-8
"""
particle optimisation
=========================

Basic demonstration of particle optimisation, speatra matching example.

author: O. Jackson, 03/2025
"""
# %%
# imports
# -------
import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np

# %%
# Setup
# -----
# we setup the parameters of optimisation incuding our initial guess.

# - Define the range of wavelengths to be incuded in optimisation.
wl0 = torch.linspace(450, 650, 50)
k0 = 2 * torch.pi / wl0

# - constants
n_env = 1.0
r_core = 12.0


r_c = torch.tensor(r_core, requires_grad=False, dtype=torch.double)

# - initial guesses (to be optimised)
r_shell = 50.0
mat_core = 2.0 + 0.1j
mat_shell = 5.0 + 0.2j

r_s = torch.tensor(r_shell, requires_grad=True, dtype=torch.double)
eps_c = torch.tensor(mat_core**2, requires_grad=True, dtype=torch.cdouble)
eps_s = torch.tensor(mat_shell**2, requires_grad=True, dtype=torch.cdouble)


# %%
# plot efficiency spectra
# ------------------
# Calculate extinction, scattering and absorption spectra for initial guess.

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

cs = pmd.farfield.cross_sections(k0, r_c, eps_c, r_s, eps_s, n_max=4)

target = gaussian(wl0.numpy(), 532.0, 75.0) * 800

# - plot
plt.figure()
plt.plot(cs["wavelength"], cs["q_ext"].detach(), label="$Q_{ext}^{init}$")
plt.plot(cs["wavelength"], cs["q_sca"].detach(), label="$Q_{sca}^{init}$")
plt.plot(cs["wavelength"], cs["q_abs"].detach(), label="$Q_{abs}^{init}$")
plt.plot(cs["wavelength"], target, label="$Q_{sca}^{target}$")

plt.xlabel("wavelength (nm)")
plt.ylabel("Efficiency")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# optimiation config
# ------------------
# get parameter tensors ready for optimisation
target = torch.tensor(target, requires_grad=False)

# - define optimister

optimizer = torch.optim.LBFGS([r_s, eps_c, eps_s], lr=0.1, max_iter=15, history_size=7)

# %%
# create optimiation loop
# ------------------
# define losses and create optimiation loop

# - define loss function


def loss_function(loss_1, k0, r_c, eps_c, r_s, eps_s):
    if r_c.item() > r_s.item():
        loss_1 = loss_1 + 0.5
    else:
        loss_1 = 0

    penaty_loss = (
        torch.nn.functional.relu(-1*r_c) 
        + torch.nn.functional.relu(-1*r_s)
        + torch.nn.functional.relu(-1*eps_c.real) + torch.nn.functional.relu(-1*eps_c.imag)
        + torch.nn.functional.relu(-1*eps_s.real) + torch.nn.functional.relu(-1*eps_s.imag)
        + loss_1
        )

    current_iter = pmd.farfield.cross_sections(k0, r_c, eps_c, r_s, eps_s, n_max=4)["q_ext"]
    loss = torch.nn.functional.mse_loss(current_iter, target) + penaty_loss
    return loss


# - create optimiation loop
loss_1 = 0
max_iter = 35
losses = []  # Array to store loss data


def closure():
    optimizer.zero_grad()  # Reset gradients

    loss = loss_function(loss_1, k0, r_c, eps_c, r_s, eps_s)  # get loss for current params

    loss.backward()  # Compute gradients
    return loss


for o in range(max_iter + 1):
    loss = optimizer.step(closure)  # LBFGS requires closure

    losses.append(loss.item())  # Store loss value

    if o % 5 == 0:
        print(o, loss.item())


# %%
# view optimised results

# - plot optimised spectra against target spectra
cs_opt = pmd.farfield.cross_sections(k0, r_c, eps_c, r_s, eps_s, n_max=4)

print("final:", [f"{d.detach().numpy():.3f}" for d in [r_c, r_s, eps_c**0.5, eps_s**0.5]])

plt.figure()
plt.plot(cs_opt["wavelength"], cs_opt["q_sca"].detach(), label="$Q_{sca}^{optim}$")
plt.plot(cs_opt["wavelength"], cs_opt["q_abs"].detach(), label="$Q_{abs}^{optim}$", linestyle = "--")
plt.plot(cs_opt["wavelength"], cs_opt["q_ext"].detach(), label="$Q_{ext}^{optim}$", linestyle = "--")
plt.plot(cs_opt["wavelength"], target, label="$Q_{sca}^{target}$")
plt.xlabel("wavelength (nm)")
plt.ylabel("Efficiency")
plt.legend()
plt.tight_layout()
plt.show()

# %%
