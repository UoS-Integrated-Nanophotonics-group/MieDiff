# encoding: utf-8
"""
Automatic differentiation
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
    mat_env=n_env,
    r_core=r_core,
    mat_core=mat_core,
    r_shell=r_shell,
    mat_shell=mat_shell,
)
print(p)

# %%
# gradients with respect to wavelength
# ------------------------------------
# Calculate the gradients of the extinction wrt the input wavelengths

cs = p.get_cross_sections(k0)
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
# -------------------------------------
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

# which Mie order to evaluate (Note: supports vectorization)
n_max = 2


# %%
# gradient wrt abs. of Mie coefficient
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# How to calculate gradient wrt magnitude of Mie coefficient.
# May be useful to suppress or maximize a specific Mie mode.

mie_coef_result = pmd.multishell.mie_coefficients(
    k0=k0,
    r_c=r_c,
    eps_c=n_c**2,
    r_s=r_s,
    eps_s=n_s**2,
    eps_env=1.0,
    n_max=n_max,
)
a_n = mie_coef_result["a_n"]

abs_a_n = torch.abs(a_n[-1, :])  # evalulate last available order
abs_a_n.backward(retain_graph=True)

print("|a_2|:", abs_a_n)
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

mie_coef_result = pmd.multishell.mie_coefficients(
    k0=k0,
    r_c=r_c,
    eps_c=n_c**2,
    r_s=r_s,
    eps_s=n_s**2,
    eps_env=1.0,
    n_max=n_max,
)
# use first particle and first wavelength. Coeff. shape is (n_mie, n_particle, n_k0)
b_n = mie_coef_result["b_n"][:, 0, 0]

# evaluate real and imag part separately of Mie coefficient (of highest order)
grad_bn_real = torch.autograd.grad(outputs=b_n[-1].real, inputs=r_s, retain_graph=True)[
    0
]
grad_bn_imag = torch.autograd.grad(outputs=b_n[-1].imag, inputs=r_s, retain_graph=True)[
    0
]

print("b_2", b_n[-1])
print("grad:", "Re:", grad_bn_real, "Im:", grad_bn_imag)


# %%
# Autodiff for internal multilayer nearfields
# -------------------------------------------
# Demonstrate gradients of the internal field in the 2nd layer 
# of a 3-layer sphere.

wl0 = torch.as_tensor([700.0], dtype=torch.float64)  # nm
k0 = 2 * torch.pi / wl0
n_env = 1.2

# define a 3-layer sphere
r_layers = torch.tensor([45.0, 80.0, 120.0], dtype=torch.float64)
n_layers = torch.tensor(
    [2.0 + 0.05j, 1.7 + 0.0j, 1.35 + 0.02j], dtype=torch.complex128
)
eps_layers = n_layers**2
eps_layers.requires_grad = True

# probe pos. inside layer #2
r_probe = torch.tensor([[60.0, 0.0, 0.0]], dtype=torch.float64)

res_nf = pmd.multishell.nearfields(
    k0=k0,
    r_probe=r_probe,
    r_layers=r_layers,
    eps_layers=eps_layers,
    eps_env=n_env**2,
)

E_t = res_nf["E_t"][0, 0, 0]
I_t = (E_t.real**2 + E_t.imag**2).sum()
I_t.backward()

print("Internal |E|^2 at r=60 nm:", I_t.detach().item())
print("grad wrt eps_layers:", eps_layers.grad)
