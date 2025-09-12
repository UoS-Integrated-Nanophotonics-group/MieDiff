# %%

import time
import matplotlib.pyplot as plt
import torch
import pymiediff as pmd


# %%
n = torch.arange(1, 10)
z = torch.rand((5, 3, 1), dtype=torch.complex64)

j_old = pmd.special.sph_jn_torch_via_rec(n, z)
j_new = pmd.special.sph_jn_torch(n, z)

dj_old = pmd.special.f_der_torch(n, z, j_new)

print(z.shape)
print(j_old.shape, j_new.shape)
print(dj_old.shape, j_new.shape)
# %%
f_n = j_new
n_max = int(n.max())
assert n_max >= 0

print(f_n.shape, z.shape)

f_n = torch.atleast_1d(f_n)
if z.dim() < f_n.dim():
    z = torch.atleast_1d(z).unsqueeze(-1)  # add order dimension
n_list = torch.arange(n_max + 1, device=z.device).broadcast_to(f_n.shape)

df = torch.zeros_like(f_n)
print(f_n.shape, z.shape, n_list.shape, df.shape)

df[..., 0] = -f_n[..., 1]
df[..., 1:] = f_n[..., :-1] - ((n_list[..., 1:] + 1) / z) * f_n[..., 1:]


# %%
# setup
# -----
# we setup the particle dimension and materials as well as the environemnt.
# This is then wrapped up in an instance of `Particle`.

# - config
wl0 = torch.linspace(500, 900, 500)
k0 = 2 * torch.pi / wl0

r_core = 460.0
r_shell = None  # r_core+40.0
mat_core = pmd.materials.MatDatabase("Ag")
# mat_core = pmd.materials.MatConstant(10 + 10j)
mat_shell = None  # pmd.materials.MatDatabase("Si")
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

# efficiency spectra
# ------------------
# Calculate extinction, scattering and absorption spectra


t0 = time.time()
cs_tr = p.get_cross_sections(k0, backend="torch", precision="double")  # , n_max=5)
# print(cs_tr['q_ext_multipoles'][1,:,0])
t1 = time.time()
cs_sc = p.get_cross_sections(k0, backend="scipy")
t2 = time.time()
print("time total =", t2 - t0)
print("time scipy =", t2 - t1)
print("time torch =", t1 - t0)
print("speedup torch =", (t2 - t1)/(t1 - t0))


# - plot
plt.figure()
plt.plot(cs_tr["wavelength"], cs_tr["q_ext"], color="C0", label="$Q_{ext} - torch$")
plt.plot(cs_tr["wavelength"], cs_tr["q_sca"], color="C1", label="$Q_{sca}$")
plt.plot(cs_tr["wavelength"], cs_tr["q_abs"], color="C2", label="$Q_{abs}$")


plt.plot(
    cs_sc["wavelength"],
    cs_sc["q_ext"],
    color="C0",
    label="$Q_{ext} - scipy$",
    dashes=[2, 2],
    lw=2.5,
)
plt.plot(
    cs_sc["wavelength"],
    cs_sc["q_sca"],
    color="C1",
    label="$Q_{sca}$",
    dashes=[2, 2],
    lw=2.5,
)
plt.plot(
    cs_sc["wavelength"],
    cs_sc["q_abs"],
    color="C2",
    label="$Q_{abs}$",
    dashes=[2, 2],
    lw=2.5,
)
plt.xlabel("wavelength (nm)")
plt.ylabel("Efficiency")
plt.legend()
plt.tight_layout()
# plt.savefig("ex_01a.svg", dpi=300)
plt.show()

# %%
# import unittest
# import torch
# import numpy as np
# from scipy.special import spherical_jn


# # %%
# class TestSphJnCF(unittest.TestCase):
#     def test_compare_with_scipy_real(self):
#         # test on a grid of real z values
#         z_vals = np.linspace(0, 20, 50)  # 50 test points
#         z_torch = torch.tensor(z_vals, dtype=torch.complex128)

#         n_max = 10
#         n = torch.arange(0, n_max + 1)

#         j_torch = sph_jn_cf(n, z_torch)  # shape (50, n_max+1)
#         j_scipy = np.stack([spherical_jn(k, z_vals) for k in range(n_max + 1)], axis=-1)

#         # convert torch -> numpy
#         j_torch_np = j_torch.detach().cpu().numpy()

#         # check absolute and relative errors
#         abs_err = np.max(np.abs(j_torch_np - j_scipy))
#         rel_err = np.max(np.abs((j_torch_np - j_scipy) / (j_scipy + 1e-16)))

#         print("max abs err:", abs_err)
#         print("max rel err:", rel_err)

#         self.assertTrue(abs_err < 1e-12, f"abs error too large: {abs_err}")
#         self.assertTrue(rel_err < 1e-10, f"rel error too large: {rel_err}")

#     def test_small_z_limit(self):
#         # near zero, j_n(z) ~ z^n/(2n+1)!!
#         z_vals = np.array([0.0, 1e-8, 1e-6])
#         z_torch = torch.tensor(z_vals, dtype=torch.complex128)
#         n_max = 5
#         n = torch.arange(0, n_max + 1)
#         j_torch = sph_jn_cf(n, z_torch)

#         j_expected = np.stack(
#             [spherical_jn(k, z_vals) for k in range(n_max + 1)], axis=-1
#         )
#         j_torch_np = j_torch.detach().cpu().numpy()

#         abs_err = np.max(np.abs(j_torch_np - j_expected))
#         self.assertTrue(abs_err < 1e-12, f"abs error near zero too large: {abs_err}")


# if __name__ == "__main__":
#     unittest.main()
