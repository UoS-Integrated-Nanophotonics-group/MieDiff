import unittest

import torch
import numpy as np
import pymiediff as pmd
import random
import functools

from scipy.special import spherical_jn, spherical_yn

# hard coded equations using scipy
# ================= DO NOT CHANGE === 03 2025 ==================================
# ==============================================================================
def sph_h1n(z, n):
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def sph_h1n_der(z, n):
    return spherical_jn(n, z, derivative=True) + 1j * spherical_yn(
        n, z, derivative=True
    )


def psi(z, n):
    return z * spherical_jn(n, z)


def chi(z, n):
    return -z * spherical_yn(n, z)


def xi(z, n):
    return z * sph_h1n(z, n)


def psi_der(z, n):
    return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)


def chi_der(z, n):
    return -spherical_yn(n, z) - z * spherical_yn(n, z, derivative=True)


def xi_der(z, n):
    return sph_h1n(z, n) + z * sph_h1n_der(z, n)


def An_sci(x, n, m1, m2):
    return (
        m2 * psi(m2 * x, n) * psi_der(m1 * x, n)
        - m1 * psi_der(m2 * x, n) * psi(m1 * x, n)
    ) / (
        m2 * chi(m2 * x, n) * psi_der(m1 * x, n)
        - m1 * chi_der(m2 * x, n) * psi(m1 * x, n)
    )


def Bn_sci(x, n, m1, m2):
    return (
        m2 * psi(m1 * x, n) * psi_der(m2 * x, n)
        - m1 * psi(m2 * x, n) * psi_der(m1 * x, n)
    ) / (
        m2 * chi_der(m2 * x, n) * psi(m1 * x, n)
        - m1 * psi_der(m1 * x, n) * chi(m2 * x, n)
    )


def an_sci(x, y, n, m1, m2):
    return (
        psi(y, n) * (psi_der(m2 * y, n) - An_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * psi_der(y, n) * (psi(m2 * y, n) - An_sci(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        xi(y, n) * (psi_der(m2 * y, n) - An_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * xi_der(y, n) * (psi(m2 * y, n) - An_sci(x, n, m1, m2) * chi(m2 * y, n))
    )


def bn_sci(x, y, n, m1, m2):
    return (
        m2
        * psi(y, n)
        * (psi_der(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - psi_der(y, n) * (psi(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        m2 * xi(y, n) * (psi_der(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - xi_der(y, n) * (psi(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi(m2 * y, n))
    )
# ==============================================================================

# possible TODO - replace this with a test using Treams
class TestCoefficientsForwards(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        dtype_complex = torch.cdouble

        n_max = 5

        self.n = torch.arange(1, n_max + 1).unsqueeze(0)

        x_y_res = 100

        # k0 = torch.linspace(0.0001,

        self.x = torch.linspace(0.01, 8.0, x_y_res, dtype=dtype_complex).unsqueeze(1)
        self.y = torch.linspace(2.0, 10.0, x_y_res, dtype=dtype_complex).unsqueeze(1)

        n_env = torch.tensor(1.0, dtype=dtype_complex)
        n_core = torch.tensor(random.uniform(0.1, 4.0) + 1j*random.uniform(0.01, 1.0), dtype=dtype_complex)
        n_shell = torch.tensor(random.uniform(0.1, 4.0) + 1j*random.uniform(0.01, 1.0), dtype=dtype_complex)

        self.m1 = torch.broadcast_to(torch.atleast_1d(n_core / n_env).unsqueeze(1), self.x.shape)
        self.m2 = torch.broadcast_to(torch.atleast_1d(n_shell / n_env).unsqueeze(1), self.x.shape)

    def tearDown(self):
        pass

    def test_forward(self):
        function_sets = [
            (
                pmd.coreshell.an,
                an_sci,
                {"x": self.x, "y": self.y, "n": self.n, "m1": self.m1, "m2": self.m2},
            ),
            (
                pmd.coreshell.bn,
                bn_sci,
                {"x": self.x, "y": self.y, "n": self.n, "m1": self.m1, "m2": self.m2},
            ),
            (
                pmd.coreshell._An,
                An_sci,
                {"x": self.x, "n": self.n, "m1": self.m1, "m2": self.m2},
            ),
            (
                pmd.coreshell._Bn,
                Bn_sci,
                {"x": self.x, "n": self.n, "m1": self.m1, "m2": self.m2},
            ),
        ]

        for func_ad, func_scipy, kwargs in function_sets:
            if self.verbose:
                print("test vs scipy: ", func_ad)

            result_ad = func_ad(**kwargs)

            kwargs_np = dict()
            for k in kwargs:
                kwargs_np[k] = kwargs[k].detach().cpu().numpy()
            result_scipy = torch.as_tensor(func_scipy(**kwargs_np))

            torch.testing.assert_close(result_scipy, result_ad)

# possible TODO - get this working consistently
# class TestCoefficientsBackward(unittest.TestCase):

#     def setUp(self):
#         self.verbose = False
#         dtype_complex = torch.cdouble

#         n_max = 5

#         self.n = torch.arange(1, n_max + 1).unsqueeze(0).contiguous()

#         x_y_res = 100

#         # k0 = torch.linspace(0.0001,

#         self.x = torch.linspace(0.5, 8.0, x_y_res, dtype=dtype_complex, requires_grad=True).unsqueeze(1).contiguous()
#         self.y = torch.linspace(2.0, 10.0, x_y_res, dtype=dtype_complex, requires_grad=True).unsqueeze(1).contiguous()

#         n_env = torch.tensor(1.0, dtype=dtype_complex).contiguous()
#         n_core = torch.tensor(random.uniform(0.1, 4.0) + 1j*random.uniform(0.01, 1.0), dtype=dtype_complex, requires_grad=True).contiguous()
#         n_shell = torch.tensor(random.uniform(0.1, 4.0) + 1j*random.uniform(0.01, 1.0), dtype=dtype_complex, requires_grad=True).contiguous()

#         self.m1 = torch.broadcast_to(torch.atleast_1d(n_core / n_env).unsqueeze(1), self.x.shape).contiguous()
#         self.m2 = torch.broadcast_to(torch.atleast_1d(n_shell / n_env).unsqueeze(1), self.x.shape).contiguous()

#     def test_backwards_an(self):
#         self.assertTrue(torch.autograd.gradcheck(pmd.coreshell.an, (self.x, self.y, self.n, self.m1, self.m2), eps=0.01, atol=0.1, rtol=0.1))

#     def test_backwards_bn(self):
#         self.assertTrue(torch.autograd.gradcheck(pmd.coreshell.bn, (self.x, self.y, self.n, self.m1, self.m2), eps=0.01, atol=0.1, rtol=0.1))

#     def test_backwards_An(self):
#         self.assertTrue(torch.autograd.gradcheck(pmd.coreshell._An, (self.x, self.n, self.m1, self.m2), eps=0.01, atol=0.1, rtol=0.1))

#     def test_backwards_Bn(self):
#         self.assertTrue(torch.autograd.gradcheck(pmd.coreshell._Bn, (self.x, self.n, self.m1, self.m2), eps=0.01, atol=0.1, rtol=0.1))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
