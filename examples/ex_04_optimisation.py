# encoding: utf-8
"""
particle optimisation
=========================

Basic demonstration of unconstained particle optimisation, speatra matching example.

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
# setup figure of merit
# -------

# - define the range of wavelengths to be incuded in optimisation.
wl0 = torch.linspace(200, 600, 100)
k0 = 2 * torch.pi / wl0


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


plt.figure()
target = gaussian(wl0.numpy(), 350.0, 100.0) * 800

plt.plot(wl0, target, label="Target spetra.")
plt.xlabel("$\lambda$ (nm)")
plt.legend()
plt.show()

target_tensor = torch.tensor(target)


# %%
# setup particle
# -------

# - constants
n_env = 1.0


# - initial guesses for pre-optimiation

num = 50

n_core_real_min, n_core_real_max = 0.5, 15.0
n_core_imag_min, n_core_imag_max = 0.01, 2.0
n_shell_real_min, n_shell_real_max = 0.5, 15.0
n_shell_imag_min, n_shell_imag_max = 0.01, 2.0

r_core_min, r_core_max = 10.0, 50.0
r_shell_min, r_shell_max = 10.0, 100.0


# n_core_array = np.random.uniform(low=0.5, high=15.0, size=num) + np.random.uniform(low=0.01, high=2.0, size=num)*1j
# n_shell_array = np.random.uniform(low=0.5, high=15.0, size=num) + np.random.uniform(low=0.01, high=2.0, size=num)*1j
# r_core_arr = np.random.uniform(low=1.0, high=50.0, size=num)
# r_shell_arr = r_core_arr + np.random.uniform(low=1.0, high=50.0, size=num)


# # %%
# # plot efficiency spectra
# # -------
# # Calculate fom of initial guesses to find starting point.
# plt.figure()
# best = 1e5, 0
# for i in range(num):
#     cs = pmd.farfield.cross_sections(k0, r_core_arr[i], (n_core_array[i])**2, r_shell_arr[i], (n_shell_array[i])**2, eps_env= 1.44)
#     plt.plot(cs["wavelength"], cs["q_sca"].detach())
#     pre_loss = torch.nn.functional.mse_loss(target_tensor, cs["q_sca"])
#     if best[0] > pre_loss:
#         best = pre_loss, i
#         print("new best loss: ", pre_loss.item())

# plt.xlabel("wavelength (nm)")
# plt.ylabel("Efficiency")
# plt.tight_layout()
# plt.show()


# # %%
# # plot results of pre-optimiation
# # -------
# print(f"best fom: {best[0]}, with rc: {r_core_arr[best[1]]}, rs: {r_shell_arr[best[1]]}")

# plt.figure()
# cs = pmd.farfield.cross_sections(k0, r_core_arr[best[1]], (n_core_array[i])**2, r_shell_arr[i], (n_shell_array[i])**2)
# plt.plot(cs["wavelength"], cs["q_sca"], label="$Q_{sca}$")
# plt.plot(cs["wavelength"], target_tensor, label="$Q_{sca}^{target}$")
# plt.legend()
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Efficiency")
# plt.tight_layout()
# plt.show()


# %%
# optimiation config
# ------------------
# - get parameter tensors ready for optimisation

r_c = torch.tensor(np.random.random(), requires_grad=True, dtype=torch.double)
r_s = torch.tensor(np.random.random(), requires_grad=True, dtype=torch.double)

n_c = torch.tensor(
    np.random.random() + np.random.random() * 1j,
    requires_grad=True,
    dtype=torch.cdouble,
)
n_s = torch.tensor(
    np.random.random() + np.random.random() * 1j,
    requires_grad=True,
    dtype=torch.cdouble,
)

# - define optimister

optimizer = torch.optim.LBFGS([r_c, n_c, r_s, n_s], lr=0.9, max_iter=5, history_size=7)


# %%
# optimiation loop
# ------------------
# define losses and create optimiation loop

max_iter = 100
losses = []  # Array to store loss data


def scale_back(normalized_value, min_value, max_value):
    return (
        normalized_value * (max_value - min_value)
        + min_value
    )


def closure():
    optimizer.zero_grad()  # Reset gradients

    # scale parameters to make physical
    args = (
        k0,
        scale_back(r_c, r_core_min, r_core_max),
        (
            scale_back(n_c.real, n_core_real_min, n_core_real_max)
            + 1j * scale_back(n_c.imag, n_core_imag_min, n_core_imag_max)
        )** 2,
        scale_back(r_c, r_core_min, r_core_max) + scale_back(r_s, r_shell_min, r_shell_max),
        (
            scale_back(n_s.real, n_shell_real_min, n_shell_real_max)
            + 1j * scale_back(n_s.imag, n_shell_imag_min, n_shell_imag_max)
        )** 2,
    )

    iteration_n = pmd.farfield.cross_sections(*args)["q_sca"]
    loss = torch.nn.functional.mse_loss(target_tensor, iteration_n)

    loss.backward()  # Compute gradients
    return loss


for o in range(max_iter + 1):

    # r_c = torch.nn.functional.sigmoid(r_c)

    loss = optimizer.step(closure)  # LBFGS requires closure

    losses.append(loss.item())  # Store loss value

    if o % 5 == 0:
        print(o, loss.item())
        print(r_c.item(), r_s.item())
        print(n_c.item(), n_s.item())


# %%
# view optimised results

# - plot optimised spectra against target spectra

args = (
    scale_back(r_c, r_core_min, r_core_max),
    (
        scale_back(n_c.real, n_core_real_min, n_core_real_max)
        + 1j * scale_back(n_c.imag, n_core_imag_min, n_core_imag_max)
    )
    ** 2,
    scale_back(r_c, r_core_min, r_core_max) + scale_back(r_s, r_shell_min, r_shell_max),
    (
        scale_back(n_s.real, n_shell_real_min, n_shell_real_max)
        + 1j * scale_back(n_s.imag, n_shell_imag_min, n_shell_imag_max)
    )
    ** 2,
)


cs_opt = pmd.farfield.cross_sections(k0, *args)
print("final:", [f"{d.detach().numpy()}" for d in args])
plt.figure()
plt.plot(cs_opt["wavelength"], cs_opt["q_sca"].detach(), label="$Q_{sca}^{optim}$")
plt.plot(
    cs_opt["wavelength"], target_tensor, label="$Q_{sca}^{target}$", linestyle="--"
)
plt.xlabel("wavelength (nm)")
plt.ylabel("Efficiency")
plt.legend()
plt.tight_layout()
plt.show()

# %%
