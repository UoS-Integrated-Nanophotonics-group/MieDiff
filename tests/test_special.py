# encoding=utf-8
# %%
import unittest

import numpy as np
import torch
from scipy.special import spherical_jn, spherical_yn

import pymiediff as pmd


class TestBackward(unittest.TestCase):
    def test_scipy(self):
        funclist = [pmd.special.sph_yn, pmd.special.sph_jn]
        for func in funclist:
            n = torch.tensor(5)
            z1 = torch.linspace(1, 10, 10) + 1j * torch.linspace(1, 10, 10)
            z1.requires_grad = True
            torch.autograd.gradcheck(func, (n, z1), eps=1e-2)

    def test_torch(self):
        funclist = [pmd.special.sph_yn_torch, pmd.special.sph_jn_torch]
        for func in funclist:
            n = torch.tensor(5)
            z1 = torch.linspace(1, 10, 10) + 1j * torch.linspace(1, 10, 10)
            z1.requires_grad = True
            torch.autograd.gradcheck(func, (n, z1), eps=1e-2)


class TestForward(unittest.TestCase):
    def test_jn_scipy(self):
        # test on a grid of real z values
        z_vals = np.linspace(0, 20, 50)  # 50 test points
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)

        n_max = 10
        n = torch.arange(0, n_max + 1)

        j_torch = pmd.special.sph_jn(n_max, z_torch)  # shape (n_max+1, 50)
        j_scipy = np.stack([spherical_jn(k, z_vals) for k in n.numpy()], axis=0)

        # convert torch -> numpy
        j_torch_np = j_torch.detach().cpu().numpy()

        # check relative errors
        rel_err = np.max(np.abs((j_torch_np - j_scipy) / (j_scipy + 1e-16)))

        self.assertTrue(rel_err < 1e-7, f"jn rel error too large: {rel_err}")

    def test_jn_torch(self):
        # test on a grid of real z values
        z_vals = np.linspace(0, 20, 50)  # 50 test points
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)

        n_max = 10
        n = torch.arange(0, n_max + 1)

        j_torch = pmd.special.sph_jn_torch(n_max, z_torch)  # shape (n_max+1, 50)
        j_scipy = np.stack([spherical_jn(k, z_vals) for k in n.numpy()], axis=0)

        # convert torch -> numpy
        j_torch_np = j_torch.detach().cpu().numpy()

        # check relative errors
        rel_err = np.max(np.abs((j_torch_np - j_scipy) / (j_scipy + 1e-16)))

        self.assertTrue(rel_err < 1e-7, f"jn rel error too large: {rel_err}")

    def test_jn_torch_small_z_limit(self):
        # near zero, j_n(z) ~ z^n/(2n+1)!!
        z_vals = np.array([0.0, 1e-16, 1e-8, 1e-6, 1e-4, 1.0])
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)
        n_max = 5
        n = torch.arange(0, n_max + 1)
        j_torch = pmd.special.sph_jn_torch(n_max, z_torch)

        j_expected = np.stack(
            [spherical_jn(k, z_vals) for k in range(n_max + 1)], axis=0
        )
        j_torch_np = j_torch.detach().cpu().numpy()

        abs_err = np.max(np.abs(j_torch_np - j_expected))
        self.assertTrue(abs_err < 1e-7, f"jn abs error near zero too large: {abs_err}")

    def test_yn_scipy(self):
        # test on a grid of real z values
        z_vals = np.linspace(0.1, 20, 50)  # 50 test points
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)

        n_max = 10
        n = torch.arange(0, n_max + 1)

        y_torch = pmd.special.sph_yn(n_max, z_torch)  # shape (n_max+1, 50)
        y_scipy = np.stack([spherical_yn(k, z_vals) for k in n.numpy()], axis=0)

        # convert torch -> numpy
        y_torch_np = y_torch.detach().cpu().numpy()

        # check relative errors
        rel_err = np.max(np.abs((y_torch_np - y_scipy) / (y_scipy + 1e-16)))

        self.assertTrue(rel_err < 1e-7, f"yn rel error too large: {rel_err}")

    def test_yn_torch(self):
        # test on a grid of real z values
        z_vals = np.linspace(0.1, 20, 50)  # 50 test points
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)

        n_max = 10
        n = torch.arange(0, n_max + 1)

        y_torch = pmd.special.sph_yn_torch(n_max, z_torch)  # shape (n_max+1, 50)
        y_scipy = np.stack([spherical_yn(k, z_vals) for k in n.numpy()], axis=0)

        # convert torch -> numpy
        y_torch_np = y_torch.detach().cpu().numpy()

        # check relative errors
        rel_err = np.max(np.abs((y_torch_np - y_scipy) / (y_scipy + 1e-16)))

        self.assertTrue(rel_err < 1e-7, f"yn rel error too large: {rel_err}")
    

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
