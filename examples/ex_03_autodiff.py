# encoding: utf-8
"""
automatic differentiation
=========================

Basic demonstration of AutoDiff with pymiediff

author: P. Wiecha, 03/2025
"""
# %%
# imports
# -------

import matplotlib.pyplot as plt
import torch
import pymiediff as pmd

backend = "torch"  # "scipy" or "torch"

# %%
# setup
# -----
# we setup the particle dimension and materials as well as the environemnt.
# This is then wrapped up in an instance of `Particle`.
#
# For all input parameter that we later want to calculate gradients, we
# set `requires_grad = True`

# - config
wl0 = torch.linspace(500, 1000, 50)
wl0.requires_grad = True
k0 = 2 * torch.pi / wl0


r_core = torch.as_tensor(70.0)
r_core.requires_grad = True

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
# gradients with respect to wavelength
# ------------------------------------
# Calculate the gradients of the extinction wrt the input wavelengths

cs = p.get_cross_sections(k0, backend=backend)
q_ext = cs["q_ext"]

# - gradient of each Q_ext wrt the wavelength
qext_grad_wl = torch.autograd.grad(
    outputs=q_ext, inputs=wl0, grad_outputs=torch.ones_like(q_ext), retain_graph=True
)[0]
print("grad wrt wavelength:", qext_grad_wl)

plt.subplot(211)
plt.plot(wl0.detach().numpy(), q_ext.detach().numpy())
plt.ylabel(r"$Q_{ext}$", fontsize=12)

plt.subplot(212)
plt.axhline(0, dashes=[2, 2], color="k")
plt.plot(wl0.detach().numpy(), qext_grad_wl.detach().numpy(), color="C1")
plt.xlabel("wavelength (nm)", fontsize=12)
plt.ylabel(r"$\partial Q_{ext} \, /\, \partial \lambda_0$", fontsize=12)
# plt.savefig("ex_03a.svg", dpi=300)
plt.show()

# %%
# gradients with respect to core radius
# ------------------------------------
# Calculate the gradients of the extinction wrt the particle core radius.
#
# Note that reverse mode autodiff requires one backwards pass per output scalar.

# - gradients of each Q_ext (every wavelength) wrt core radius
qext_grad_rcore = []
for q_wl in q_ext:
    qext_grad_rcore.append(
        torch.autograd.grad(
            outputs=q_wl, inputs=len(q_ext) * [r_core], retain_graph=True
        )[0]
    )
qext_grad_rcore = torch.stack(qext_grad_rcore)
print("grad wrt core radius:", qext_grad_rcore)

plt.subplot(211)
plt.plot(wl0.detach().numpy(), q_ext.detach().numpy())
plt.ylabel(r"$Q_{ext}$", fontsize=12)

plt.subplot(212)
plt.axhline(0, dashes=[2, 2], color="k")
plt.plot(wl0.detach().numpy(), qext_grad_rcore.detach().numpy(), color="C1")
plt.xlabel("wavelength (nm)", fontsize=12)
plt.ylabel(r"$\partial Q_{ext} \, /\, \partial r_{core}$", fontsize=12)
# plt.savefig("ex_03b.svg", dpi=300)
plt.show()


# %%
# Gradients of Mie coefficients
# -----------------------------
# Using the lower level functions, we can also calculate gradients of Mie coefficients

# - some radii and ref.index particle config for this demo
wl0 = [500.0]  # nm
k0 = 2 * torch.pi / torch.as_tensor(wl0)
n_c = torch.as_tensor(3.0)
n_s = torch.as_tensor(4.0)
r_c = torch.as_tensor(110.0)  # nm
r_s = torch.as_tensor(130.0)  # nm

r_s.requires_grad = True

# - prepare evaluation of Mie coefficients
n_max = 2  # which Mie order to evaluate (Note: supports vectorization)
x = k0 * r_c
y = k0 * r_s
m_c = n_c / n_env
m_s = n_s / n_env

# %%
# gradient wrt abs. of Mie coefficient
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# How to calculate gradient wrt magnitude of Mie coefficient.
# May be useful to suppress or maximize a specific Mie mode.

a_n, b_n = pmd.coreshell.ab(x, y, n_max, m_c, m_s)

abs_a_n = torch.abs(a_n[:, -1])  # evalulate last available order
abs_a_n.backward(retain_graph=True)

print("|a_n|:", abs_a_n)
print(
    "grad:",
    r_s.grad,
    ": change the shell radius into this direction will reduce |a_n|.",
)

# %%
# gradient wrt complex Mie coefficient
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Calculate gradients of complex values requires to evaluate real and
# imaginary parts separately. The respective partial derivatives are
# the real and imag part of the gradient.

a_n, b_n = pmd.coreshell.ab(x, y, n_max, m_c, m_s)

# evaluate real and imag part separately of Mie coefficient (of highest order)
grad_bn_real = torch.autograd.grad(
    outputs=b_n[0, -1].real, inputs=r_s, retain_graph=True
)[0]
grad_bn_imag = torch.autograd.grad(
    outputs=b_n[0, -1].imag, inputs=r_s, retain_graph=True
)[0]

print("b_n", b_n)
print("grad:", "Re:", grad_bn_real, "Im:", grad_bn_imag)
