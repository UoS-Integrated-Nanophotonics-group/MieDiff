# encoding: utf-8
"""
particle optimisation
=====================

Demonstration of particle optimisation via gradient descent.
Farfield cross sections are optimiatised to fit a guassian curve centered
at 600.0nm.

Core and shell refractive indices and radii are optimised, with the materials
limited to dispersionless dielectrics.

We optimize a large number of initial guesses concurrently, which avoids that
a single solution gets stuck in a local minimum.

author: O. Jackson, P. Wiecha 03/2025
"""
# %%
# imports
# -------
import time
import matplotlib.pyplot as plt
import pymiediff as pmd
import torch
import numpy as np

backend = "torch"

# %%
# setup optimiation target
# ------------------------

# - define the range of wavelengths to be incuded in optimisation.
N_wl = 21
wl0 = torch.linspace(400, 800, N_wl)
k0 = 2 * torch.pi / wl0


# - for this example we target a gaussian like spectra centered at 600.0nm
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
# plt.savefig("ex_04a.svg", dpi=300)
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


def params_to_physical(r_opt, n_opt):
    """converts normalised parameters to physical

    Args:
        r_opt (torch.Tensor): normalised radii
        n_opt (torch.Tensor): normalised materials

    Returns:
        torch.Tensor: physical parameters
    """

    # constrain optimization internally to physical limits
    # sigmoid: convert to [0, 1], then renormalize to physical limits
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

    return r_c, n_c**2, r_s, n_s**2


# %%
# random initialization
# ---------------------
# we use PyMieDiff's vectorization capabilities to run the optimization of
# many random initial guesses in parallel.

# number of random guesses to make.
num_guesses = 100

# 2 size parameters (radius of core and thickness of shell)
# 4 material parameters: real and imag parts of constant ref. indices
r_opt_arr = torch.normal(0, 1, (2, num_guesses))
n_opt_arr = torch.normal(0, 1, (4, num_guesses))
r_opt_arr.requires_grad = True
n_opt_arr.requires_grad = True


# %%
# optimisation loop
# ------------------
# define losses, create and run optimization loop. In this example
# adam optimizer is used, but the example is written such that it is
# ready to be used with LBFGS instead (requiring a "closure").

max_iter = 50


# - define optimiser and hyperparameters
optimizer = torch.optim.AdamW(
    [r_opt_arr, n_opt_arr],
    lr=0.2,
)
# - alternative optimizer: LBFGS
# optimizer = torch.optim.LBFGS(
#     [r_opt_arr, n_opt_arr], lr=0.2, max_iter=10, history_size=7
# )


# - helper for batched forward pass (many particles)
def eval_batch(r_opt_arr, n_opt_arr):
    r_c, eps_c, r_s, eps_s = params_to_physical(r_opt_arr, n_opt_arr)

    # spectrally expand the permittivities
    eps_c = eps_c.unsqueeze(1).unsqueeze(1).broadcast_to(num_guesses, N_wl, 1)
    eps_s = eps_s.unsqueeze(1).unsqueeze(1).broadcast_to(num_guesses, N_wl, 1)

    # evaluate Mie
    result_mie = pmd.coreshell.cross_sections(
        k0.unsqueeze(0), r_c, eps_c, r_s, eps_s, backend=backend
    )["q_sca"]

    # get loss, MSE comparing target with current spectra
    losses = torch.mean(torch.abs(target_tensor.unsqueeze(0) - result_mie) ** 2, dim=1)

    return losses


# - required for LFBGS: closure (LFBGS calls f several times per iteration)
def closure():
    optimizer.zero_grad()  # Reset gradients

    losses = eval_batch(r_opt_arr, n_opt_arr)
    loss = torch.mean(losses)

    loss.backward()
    return loss


# - main loop
start_time = time.time()
loss_hist = []  # Array to store loss data
for o in range(max_iter + 1):

    loss = optimizer.step(closure)  # LBFGS requires closure
    all_losses = eval_batch(r_opt_arr, n_opt_arr)
    loss_hist.append(loss.item())  # Store loss value

    if o % 1 == 0:
        i_best = torch.argmin(all_losses)
        r_c, eps_c, r_s, eps_s = params_to_physical(r_opt_arr, n_opt_arr)
        print(
            " --- iter {}: loss={:.2f}, best={:.2f}".format(
                o, loss.item(), all_losses.min().item()
            )
        )
        print(
            "     r_core, r_shell  = {:.1f}nm,     {:.1f}nm".format(
                r_c[i_best], r_s[i_best]
            )
        )
        print(
            "     n_core, n_shell  = {:.2f}, {:.2f}".format(
                torch.sqrt(eps_c[i_best]), torch.sqrt(eps_s[i_best])
            )
        )

# - finished
print(50 * "-")
t_opt = time.time() - start_time
print(
    "Optimization finished in {:.1f}s ({:.1f}s per iteration)".format(
        t_opt, t_opt / max_iter
    )
)

# %%
# optimisation results
# --------------------
# view optimised speactra and corresponding particle parameters.

# - plot optimised spectra against target spectra
wl0_eval = torch.linspace(400, 800, 200)
k0_eval = 2 * torch.pi / wl0_eval

i_best = torch.argmin(all_losses)
r_c, eps_c, r_s, eps_s = params_to_physical(r_opt_arr[:, i_best], n_opt_arr[:, i_best])

cs_opt = pmd.coreshell.cross_sections(k0_eval, r_c, eps_c, r_s, eps_s)

plt.figure(figsize=(5, 3.5))
plt.plot(cs_opt["wavelength"], cs_opt["q_sca"][0].detach(), label="$Q_{sca}^{optim}$")
plt.plot(wl0, target_tensor, label="$Q_{sca}^{target}$", linestyle="--")
plt.xlabel("wavelength (nm)")
plt.ylabel("Scattering efficiency")
plt.legend()
plt.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 2


# - print optimun parameters
print(50 * "-")
print("optimum:")
print(" r_core  = {:.1f}nm".format(r_c))
print(" r_shell = {:.1f}nm".format(r_s))
print(" n_core  = {:.2f}".format(torch.sqrt(eps_c)))
print(" n_shell = {:.2f}".format(torch.sqrt(eps_s)))
