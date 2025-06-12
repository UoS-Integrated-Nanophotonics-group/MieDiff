# encoding=utf-8
"""
wavelength and mie coefficients vectorization demo

author: P. R. Wiecha, 02/2025
"""
# %%

import torch
import matplotlib.pyplot as plt
import pymiediff as pmd


N_wl = 200
N_mie = 5

# --- prep. for vectorization
# let's define:
# - dimension 0 will be for wavelengths
# - dimension 1 will be for mie-order
# Note: not sure if this is the best choice

# add a dimension that will be used of the mie-order
wl0 = torch.linspace(400, 800, N_wl).unsqueeze(1)

# add a dimension reserved for the wavelengths
n = torch.arange(1, N_mie + 1).unsqueeze(0)

# --- config
# todo: materials should support dispersion handling
n_env = 1.0
n_core = torch.ones(wl0.shape, dtype=torch.complex64) * 4.0 + 0j
n_shell = torch.ones(wl0.shape, dtype=torch.complex64) * 2.0
r_core = 180.0
r_shell = r_core + 10.0

k0 = 2 * torch.pi / wl0

r_c = torch.tensor(r_core)
r_s = torch.tensor(r_shell)
x = k0 * r_c
y = k0 * r_s

m_c = n_core / n_env
m_s = n_shell / n_env

print("n_core shape", n_core.shape)
print("n_shell shape", n_shell.shape)
# %%
print(n.shape, y.shape)
print((n * y).shape)
print(pmd.special.Jn(n, y).shape, pmd.special.sph_jn_torch(n, y).shape)
print(pmd.special.Yn(n, y).shape, pmd.special.sph_yn_torch(n, y).shape)


test_plt1 = pmd.special.Yn(n, y)
test_plt2 = pmd.special.sph_yn_torch(n, y)
# plt.plot(test_plt1)
# plt.plot(test_plt2[...,1:], dashes=[2,2])
# %%
import time

# --- eval Mie coefficients (vectorized)
ta = time.time()
for i in range(1):
    a_n = pmd.coreshell.an(x, y, n, m_c, m_s)
    b_n = pmd.coreshell.ab(x, y, n, m_c, m_s)
t0 = time.time()
for i in range(10):
    a_n, b_n = pmd.coreshell.ab(x, y, n, m_c, m_s)
t1 = time.time()
for i in range(10):
    a_n_g, b_n_g = pmd.coreshell.ab_gpu(x, y, n, m_c, m_s)
t2 = time.time()


print("original wrapper:", (t0 - ta) * 10)
print("optimized wrapper:", t1 - t0)
print("native torch:", t2 - t1)
print("speedup cpu: x{:.2f}".format((t0 - ta) * 10 / (t2 - t1)))

if torch.cuda.is_available():
    t3a = time.time()
    for i in range(10):
        xcu = x.to("cuda")
        ycu = y.to("cuda")
        ncu = n.to("cuda")
        m_c_cu = m_c.to("cuda")
        m_s_cu = m_s.to("cuda")
        a_n_g_cu, b_n_g_cu = pmd.coreshell.ab_gpu(xcu, ycu, ncu, m_c_cu, m_s_cu)

    t3 = time.time()
    print("native torch - cuda:", t3 - t3a)
    print("speedup cuda: x{:.2f}".format((t0 - ta) * 10 / (t3 - t3a)))

# %%
plt.plot(a_n)
plt.plot(a_n_g[...,], dashes=[2, 2])


# %%

print("a_n shape", a_n.shape)
print("b_n shape", b_n.shape)

# --- eval observables
# geometric cross section
cs_geo = torch.pi * r_shell**2

# scattering efficiencies
prefactor = 2 * torch.pi / (k0**2)
cs_ext = torch.sum(prefactor * (2 * n + 1) * ((a_n + b_n).real), dim=1)
cs_sca = torch.sum(
    prefactor * (2 * n + 1) * (a_n.real**2 + a_n.imag**2 + b_n.real**2 + b_n.imag**2),
    dim=1,
)
cs_abs = cs_ext - cs_sca

qext = cs_ext / cs_geo
qabs = cs_abs / cs_geo
qsca = cs_sca / cs_geo

print("qext shape", qext.shape)
print("qsca shape", qsca.shape)


# separate multipole contributions
# here we want to keep the mie-order dimensions, no summation.
qext_e = prefactor * (2 * n + 1) * (a_n.real) / cs_geo
qsca_e = prefactor * (2 * n + 1) * (a_n.real**2 + a_n.imag**2) / cs_geo
qext_m = prefactor * (2 * n + 1) * (b_n.real) / cs_geo
qsca_m = prefactor * (2 * n + 1) * (b_n.real**2 + b_n.imag**2) / cs_geo
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


res_cs = pmd.farfield.cross_sections(
    k0=k0.squeeze(),  # vectorization is done internally
    r_c=r_c,
    eps_c=n_core.squeeze() ** 2,
    r_s=r_s,
    eps_s=n_shell.squeeze() ** 2,
    eps_env=1.0,
    backend="gpu",
)

import pymiecs as mie

res_np = mie.main.Q(
    k0.squeeze().detach().numpy(),
    r_core=r_core,
    r_shell=r_shell,
    n_core=n_core.detach().numpy()[0],
    n_shell=n_shell.detach().numpy()[0],
)


# --- plot some of it for testing
plt.plot(wl0, qext, label="ext-explicit")
plt.plot(wl0, torch.sum(qext_e + qext_m, axis=1), dashes=[1, 1], label="sum-mp")
plt.plot(wl0, res_np["qext"], label="ext-numpy")
plt.plot(wl0, res_cs["q_ext"][0].detach(), label="ext-pmd", dashes=[2, 2])

for i in n.squeeze():
    plt.plot(wl0, qext_e[:, i - 1], label=f"ext-a{i}")
    plt.plot(wl0, qext_m[:, i - 1], label=f"ext-b{i}", dashes=[2, 2])
plt.xlabel("wavelength (nm)")
plt.ylabel("extinction efficiency")
plt.legend()
plt.show()


# %% vectorize: multiple particles
x_mult = torch.stack([x, x * 0.95])
y_mult = torch.stack([y, y * 0.95])
n_mult = torch.stack([n, n])
m_c_mult = torch.stack([m_c, m_c * 0.95])
m_s_mult = torch.stack([m_s, m_s * 0.95])

a_n, b_n = pmd.coreshell.ab(x_mult, y_mult, n_mult, m_c_mult, m_s_mult)
plt.plot(a_n[1])

#%% multi partciles cross sections
r_c_mult = torch.as_tensor([r_c, r_c])
r_s_mult = torch.as_tensor([r_s, r_s])
eps_c_mult = torch.stack([n_core**2,n_core**2])
eps_s_mult = torch.stack([n_shell**2, n_shell**2])
res = pmd.farfield.cross_sections(
    k0=k0, r_c=r_c_mult, r_s=r_s_mult, eps_c=eps_c_mult, eps_s=eps_s_mult
)


# %%
r_c_mult = torch.as_tensor([150, 180])
r_s_mult = torch.as_tensor([30, 40])
n_core_mult = torch.stack([n_core, n_core])
n_shell_mult = torch.stack([n_shell, n_shell])


res_cs = pmd.farfield.cross_sections(
    k0=k0.squeeze(),  # vectorization is done internally
    r_c=r_c_mult,
    eps_c=n_core_mult**2,
    r_s=r_s_mult,
    eps_s=n_shell_mult**2,
    eps_env=1.0,
    backend="gpu",
)
plt.plot(res_cs["cs_ext"][0])
print(k0.shape)
print(res_cs["q_ext"].shape)

# # %%
# # compare multipole contributions: pymiediff vs explicit calculation
# for m, mp in enumerate(res_cs["q_ext_multipoles"]):
#     for n, order in enumerate(mp.T):
#         if n >= N_mie:
#             break
#         plt.plot(wl0, order, label=f"type{m},order{n+1}")  # pymiediff
#         if m == 0:
#             plt.plot(wl0, qext_e[:, n], label=f"ext-a{n+1}", dashes=[2, 2])
#         if m == 1:
#             plt.plot(wl0, qext_m[:, n], label=f"ext-b{n+1}", dashes=[2, 2])

# plt.legend()
# plt.show()

# # %%
# # autograd test on vectorized implementation
# r_c = torch.tensor(60.0, requires_grad=True)
# r_s = torch.tensor(100.0, requires_grad=True)
# n_c = torch.tensor(4.0, requires_grad=True)
# n_s = torch.tensor(3.0, requires_grad=True)

# res_cs = pmd.farfield.cross_sections(
#     k0=k0.squeeze(),  # vectorization is done internally
#     r_c=r_c,
#     eps_c=n_c**2,
#     r_s=r_s,
#     eps_s=n_s**2,
#     eps_env=1,
#     n_max=5,
# )


# # calc gradients wrt size and (dispersionless) ref.indices
# q_ext = res_cs["q_ext"]
# r_c_grad = torch.autograd.grad(
#     outputs=q_ext, inputs=[r_c, r_s, n_c, n_s], grad_outputs=torch.ones_like(q_ext)
# )
# print("Grad output:", r_c_grad)


# # %%
# # test gradients using autograd
# def test_func_qext(k0, r_c, n_c, r_s, n_s):
#     res_cs = pmd.farfield.cross_sections(
#         k0=k0.squeeze()[::5],  # vectorization is done internally
#         r_c=r_c,
#         eps_c=n_c**2,
#         r_s=r_s,
#         eps_s=n_s**2,
#         eps_env=1,
#         n_max=5,
#     )
#     return res_cs["q_ext"]


# # %%
# check = torch.autograd.gradcheck(
#     test_func_qext, [k0, r_c, n_c, r_s, n_s], eps=0.002, rtol=1e-2, atol=1e-3
# )
# print("autograd check: ", check)


# # %%
# # test real materials
# import pymiediff.materials as mat

# si = mat.MatDatabase("si")
# si.plot_refractive_index()

# au = mat.MatDatabase("au")
# au.plot_refractive_index()

# eps_c = si.get_epsilon(wavelength=wl0)
# eps_s = au.get_epsilon(wavelength=wl0)

# res_realmat = pmd.farfield.cross_sections(
#     k0, r_c=r_core, r_s=r_shell, eps_c=eps_c, eps_s=eps_s
# )

# plt.figure()
# plt.plot(res_realmat["wavelength"], res_realmat["q_ext"])
# plt.show()


# # %%
# N_angular = 180
# theta = torch.linspace(0.01, 2 * torch.pi - 0.01, N_angular)
# res_angSca = pmd.farfield.angular_scattering(
#     k0=k0,
#     theta=theta,
#     r_c=r_core,
#     r_s=r_shell,
#     eps_c=eps_c,
#     eps_s=eps_s,
#     eps_env=1.0,
# )

# i_wl = 5
# plt.plot(res_angSca['theta'], res_angSca['i_unpol'][i_wl])
# %%
