# encoding=utf-8
"""
wavelength and mie coefficients vectorization demo

author: P. R. Wiecha, 11/2024
"""
# %%

import torch
import numpy as np
import matplotlib.pyplot as plt
import pymiediff as pmd


Nwl = 100
n_mie = 4

# --- prep. for vectorization
# let's define:
# - dimension 0 will be for wavelengths
# - dimension 1 will be for mie-order
# Note: not sure if this is the best choice

# add a dimension that will be used of the mie-order
wl = torch.linspace(400, 800, Nwl).unsqueeze(1)

# add a dimension reserved for the wavelengths
n = torch.arange(n_mie).unsqueeze(0)

# --- config
# todo: materials should support dispersion handling
n_env = 1.0
n_core = torch.ones(wl.shape, dtype=torch.complex64) * 4.0 + 0j
n_shell = torch.ones(wl.shape, dtype=torch.complex64) * 2.0
r_core = 80.0
r_shell = r_core + 10.0

k = 2 * torch.pi / (wl / n_env)

r_c = torch.tensor(r_core)
r_s = torch.tensor(r_shell)
x = k * r_c
y = k * r_s

m1 = n_core / n_env
m2 = n_shell / n_env

print("n_core shape", n_core.shape)
print("n_shell shape", n_shell.shape)


# --- eval Mie coefficients (vectorized)
a_n = pmd.coreshell.an(x, y, n, m1, m2)
b_n = pmd.coreshell.bn(x, y, n, m1, m2)

print("a_n shape", a_n.shape)
print("b_n shape", b_n.shape)

# --- eval observables
# geometric cross section
cs_geo = torch.pi * r_shell**2

# scattering efficiencies
prefactor = 2 / (k**2 * r_shell**2)
qext = torch.sum(prefactor * (2 * n + 1) * (a_n.real + b_n.real), dim=1)
qsca = torch.sum(
    prefactor * (2 * n + 1) * (a_n.real**2 + a_n.imag**2 + b_n.real**2 + b_n.imag**2),
    dim=1,
)
qabs = qext - qsca


print("qext shape", qext.shape)
print("qsca shape", qsca.shape)


# separate multipole contributions
# here we want to keep the mie-order dimensions, no summation.
qext_e = prefactor * (2 * n + 1) * (a_n.real)
qsca_e = prefactor * (2 * n + 1) * (a_n.real**2 + a_n.imag**2)
qext_m = prefactor * (2 * n + 1) * (b_n.real)
qsca_m = prefactor * (2 * n + 1) * (b_n.real**2 + b_n.imag**2)
qabs_e = qext_e - qsca_e
qabs_m = qext_m - qsca_m

# fw / bw scattering.
# here the mie-order dim. is already summed over when we multiply
# with prefactor (containing vector k). Therefore, with squeeze
# we removes the empty dimensions (wavelength)
qback = (prefactor / 2).squeeze() * (
    torch.abs(torch.sum((2 * n + 1) * ((-1) ** n) * (a_n - b_n), dim=1)) ** 2
)
qfwd = (prefactor / 2).squeeze() * (
    torch.abs(torch.sum((2 * n + 1) * (a_n + b_n), dim=1)) ** 2
)
qratio = qback / qfwd


res_cs = pmd.coreshell.scs(
    k0=k.squeeze(),  # vectorization is done internally
    r_c=r_c,
    eps_c=n_core.squeeze() ** 2,
    r_s=r_s,
    eps_s=n_shell.squeeze() ** 2,
    eps_env=1,
    n_max=5,
)


# --- plot some of it for testing
plt.plot(wl, qext, label="full ext")
plt.plot(wl, res_cs["q_ext"], label="full ext-2", dashes=[2, 2])

for i in n.squeeze():
    plt.plot(wl, qext_e[:, i], label=f"ext-a{i}")
    plt.plot(wl, qext_m[:, i], label=f"ext-b{i}", dashes=[2, 2])
plt.xlabel("wavelength (nm)")
plt.ylabel("extinction efficiency")
plt.legend()
plt.show()


# %%
# autograd test on vectorized implementation
r_c = torch.tensor(60.0, requires_grad=True)
r_s = torch.tensor(100.0, requires_grad=True)
n_c = torch.tensor(4.0, requires_grad=True)
n_s = torch.tensor(3.0, requires_grad=True)

res_cs = pmd.coreshell.scs(
    k0=k.squeeze(),  # vectorization is done internally
    r_c=r_c,
    eps_c=n_c**2,
    r_s=r_s,
    eps_s=n_s**2,
    eps_env=1,
    n_max=5,
)


# calc gradients wrt size and (dispersionless) ref.indices
q_ext = res_cs["q_ext"]
r_c_grad = torch.autograd.grad(
    outputs=q_ext, inputs=[r_c, r_s, n_c, n_s], grad_outputs=torch.ones_like(q_ext)
)
print("Grad output:", r_c_grad)


# %%
# test gradients using autograd
def test_func_qext(k0, r_c, n_c, r_s, n_s):
    res_cs = pmd.coreshell.scs(
        k0=k.squeeze()[::5],  # vectorization is done internally
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
        n_max=5,
    )
    return res_cs["q_ext"]


check = torch.autograd.gradcheck(
    test_func_qext, [k, r_c, n_c, r_s, n_s], eps=0.002, rtol=1e-2, atol=1e-3
)