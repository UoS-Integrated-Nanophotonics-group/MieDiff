# encoding: utf-8
"""
particle optimisation
=========================

Basic demonstration of particle optimisation in the visible light range.
Farfield cross sections are optimiatised to fit a guassian curve centered 
at 600.0nm.

Core and shell refrective indexs and radii are optimised, with the materials
limited to dielectric. 

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
# setup optimiation target
# ------------------------

# - define the range of wavelengths to be incuded in optimisation.
wl0 = torch.linspace(400, 800, 21)
k0 = 2 * torch.pi / wl0

# - for this example we target a guassian like spectra centered at 600.0nm
def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )
target = gaussian(wl0.numpy(), 600.0, 60.0) * 700 + 0.5

target_tensor = torch.tensor(target)

# - we can plot the target spectra
plt.figure()
plt.plot(wl0, target, label="Target spetra.")
plt.xlabel("$\lambda$ (nm)")
plt.legend()
plt.show()


# %%
# setup particle prameter limits
# ------------------------------

# - constants
n_env = 1.0

# - set limits to particle's properties, in this example we limit to dielectric materials
lim_r = torch.as_tensor([10, 100], dtype=torch.double)
lim_n_re = torch.as_tensor([1, 4.5], dtype=torch.double)
lim_n_im = torch.as_tensor([0, 0.1], dtype=torch.double)

# %%
# normalization helper
# --------------------
# we let the optimizer work on normalized parameters which we pass through a sigmoid. 
# This is a straightforward way to implement box boundaries for the optimization variables.

def normalize_and_convert_to_physical(k0, r_opt, n_opt):
    # sigmoid: constrain to a given parameter range
    sigmoid = torch.nn.Sigmoid()
    r_c_n, d_s_n = sigmoid(r_opt)
    n_c_re_n, n_s_re_n, n_c_im_n, n_s_im_n = sigmoid(n_opt)

    # scale parameters to physical units
    # size parameters
    r_c = r_c_n * (lim_r.max() - lim_r.min()) + lim_r.min()
    d_s = d_s_n * (lim_r.max() - lim_r.min()) + lim_r.min()
    r_s = r_c + d_s
    
    # core and shell complex ref. index
    n_c = (n_c_re_n * (lim_n_re.max() - lim_n_re.min()) + lim_n_re.min()) + 1j * (
        n_c_im_n * (lim_n_im.max() - lim_n_im.min()) + lim_n_im.min()
    )
    n_s = (n_s_re_n * (lim_n_re.max() - lim_n_re.min()) + lim_n_re.min()) + 1j * (
        n_s_im_n * (lim_n_im.max() - lim_n_im.min()) + lim_n_im.min()
    )

    args = (k0, r_c, n_c**2, r_s, n_s**2)
    return args


# %%
# optimisation config
# ------------------
# random initial guesses. here we impliment a simple global 
# search to improve the gradient optimization.

pre_opt = 500
# array to hold best initial guess
best = [100.0, np.random.random(2), np.random.random(4)]

# contains 2 values: radius of core and thickness of shell
r_opt_arr = np.random.random((2, pre_opt))
# contains 4 values: real and imag parts of core and shell ref.index
n_opt_arr = np.random.random((4, pre_opt))

for i in range(pre_opt):
    args = normalize_and_convert_to_physical(
        k0,
        torch.tensor(r_opt_arr[:,i], dtype=torch.double), 
        torch.tensor(n_opt_arr[:,i], dtype=torch.double),
        )

    # evaluate Mie
    result_mie = pmd.farfield.cross_sections(*args)["q_sca"]
    # get loss, MSE comparing target with current spectra
    loss = torch.nn.functional.mse_loss(target_tensor, result_mie)
    # update best initial guess
    if loss < best[0]:
        best[0] = loss
        best[1] = r_opt_arr[:,i]
        best[2] = n_opt_arr[:,i]
        print("new best with loss:", loss.item())


# %%
# setup tensors for optimisation
# ------------------------------
# initialise tensors for optimisation using the best guess found in the
# global search. NOTE. Run from this cell down if you want to experiment  
# with optimiser hyperparametrs.

r_opt = torch.tensor(best[1], requires_grad=True, dtype=torch.double)
n_opt = torch.tensor(best[2], requires_grad=True, dtype=torch.double)

# %%
# optimisation loop
# ------------------
# define losses, create and run optimization loop. In this example a
# LBFGS optimisier is used.

# - define optimiser and hyperparameters
optimizer = torch.optim.LBFGS([r_opt, n_opt], lr=0.25, max_iter=10, history_size=7)
max_iter = 40

# for LFBGS: closure 
def closure():
    optimizer.zero_grad()  # Reset gradients
    
    # scale parameters to physical units
    args = normalize_and_convert_to_physical(k0, r_opt, n_opt)

    # evaluate Mie
    result_mie = pmd.farfield.cross_sections(*args)["q_sca"]
    loss = torch.nn.functional.mse_loss(target_tensor, result_mie)

    loss.backward()  # Compute gradients (using AutoDiff)
    return loss


# main loop
losses = []  # Array to store loss data
for o in range(max_iter + 1):

    loss = optimizer.step(closure)  # LBFGS requires closure

    losses.append(loss.item())  # Store loss value

    if o % 5 == 0:
        print(" --- iter {}: loss={:.2f}".format(o, loss.item()))
        args = normalize_and_convert_to_physical(k0, r_opt, n_opt)
        print("     r_core  = {:.1f}nm".format(args[1]))
        print("     r_shell = {:.1f}nm".format(args[3]))
        print("     n_core  = {:.2f}".format(torch.sqrt(args[2])))
        print("     n_shell = {:.2f}".format(torch.sqrt(args[4])))


# %%
# optimisation results
# --------------------
# view optimised speactra and corresponding particle parameters.

# - plot optimised spectra against target spectra
wl0_eval = torch.linspace(400, 800, 151)
k0_eval = 2 * torch.pi / wl0_eval
args = normalize_and_convert_to_physical(k0_eval, r_opt, n_opt)

cs_opt = pmd.farfield.cross_sections(*args)

plt.figure()
plt.plot(cs_opt["wavelength"], cs_opt["q_sca"].detach(), label="$Q_{sca}^{optim}$")
plt.plot(wl0, target_tensor, label="$Q_{sca}^{target}$", linestyle="--")
plt.xlabel("wavelength (nm)")
plt.ylabel("Efficiency")
plt.legend()
plt.tight_layout()
plt.show()

# - print optimun parameters
print("optimum:")
print(" r_core  = {:.1f}nm".format(args[1]))
print(" r_shell = {:.1f}nm".format(args[3]))
print(" n_core  = {:.2f}".format(torch.sqrt(args[2])))
print(" n_shell = {:.2f}".format(torch.sqrt(args[4])))

# %%
