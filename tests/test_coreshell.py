import unittest

import torch
import numpy as np
import pymiediff as pmd
import random

from scipy.special import spherical_jn, spherical_yn


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


def cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4):
    return (2 * np.pi / k**2) * (
        (2 * n1 + 1) * (a1.abs() ** 2 + b1.abs() ** 2)
        + (2 * n2 + 1) * (a2.abs() ** 2 + b2.abs() ** 2)
        + (2 * n3 + 1) * (a3.abs() ** 2 + b3.abs() ** 2)
        + (2 * n4 + 1) * (a4.abs() ** 2 + b4.abs() ** 2)
    )


class TestCoefficientsForwards(unittest.TestCase):

    def setUp(self):
        self.verbose = False

        N_pt_test = 200

        self.n = torch.tensor(5)

        wlRes = 1000
        wl = np.linspace(200, 600, wlRes)
        r_core = 12.0
        r_shell = 50.0

        n_env = 1
        n_core = 2 + 0j
        n_shell = 5 + 0j

        dtype = torch.cfloat  # torch.complex64

        k = 2 * np.pi / (wl / n_env)
        self.k = torch.tensor(k, dtype=dtype)

        self.m1 = n_core / n_env
        self.m2 = n_shell / n_env

        self.r_c = torch.tensor(r_core, dtype=dtype)
        self.r_s = torch.tensor(r_shell, dtype=dtype)

    def tearDown(self):
        pass

    def test_forward(self):
        x = self.k * self.r_c
        y = self.k * self.r_s
        function_sets = [
            (pmd.coreshell.an, an_sci, {"x" : x, "y" : y, "n" : self.n, "m1" : self.m1, "m2" : self.m2}),
            (pmd.coreshell.bn, bn_sci, {"x" : x, "y" : y, "n" : self.n, "m1" : self.m1, "m2" : self.m2}),
            (pmd.coreshell.An, An_sci, {"x" : x, "n" : self.n, "m1" : self.m1, "m2" : self.m2}),
            (pmd.coreshell.Bn, Bn_sci, {"x" : x, "n" : self.n, "m1" : self.m1, "m2" : self.m2}),
        ]

        for func_ad, func_scipy, kwargs in function_sets:
            if self.verbose:
                print("test vs scipy: ", func_ad)

            result_ad = func_ad(**kwargs)

            result_scipy = torch.as_tensor(func_scipy(**kwargs))

            torch.testing.assert_close(result_scipy, result_ad)


class TestCoefficientsBackward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

        N_pt_test = 200

        self.n = torch.tensor(5)

        wlRes = 1000
        wl = np.linspace(200, 600, wlRes)
        r_core = 12.0
        r_shell = 50.0

        n_env = 1
        n_core = 2 + 0j
        n_shell = 5 + 0j

        dtype = torch.cfloat  # torch.complex64

        k = 2 * np.pi / (wl / n_env)
        self.k = torch.tensor(k, dtype=dtype)

        self.m1 = torch.tensor(n_core / n_env)
        self.m2 = torch.tensor(n_shell / n_env)

        self.r_c = torch.tensor(r_core, dtype=dtype)
        self.r_s = torch.tensor(r_shell, dtype=dtype)





    def num_diff_ab(self, func, x, y, n, m1, m2, eps=0.00001 + 0.00001j):
        """numerical center diff for comparison to autograd"""
        dz = []

        x1 = x.conj()
        fm = func(x1 - eps, y, n, m1, m2)
        fp = func(x1 + eps, y, n, m1, m2)
        dz.append((fp - fm) / (2 * eps))

        y1 = y.conj()
        fm = func(x, y1 - eps, n, m1, m2)
        fp = func(x, y1 + eps, n, m1, m2)
        dz.append((fp - fm) / (2 * eps))

        m11 = m1.conj()
        fm = func(x, y, n, m11 - eps, m2)
        fp = func(x, y, n, m11 + eps, m2)
        dz.append((fp - fm) / (2 * eps))

        m12 = m2.conj()
        fm = func(x1, y, n, m1, m12 - eps)
        fp = func(x1, y, n, m1, m12 + eps)
        dz.append((fp - fm) / (2 * eps))

        return dz


    def num_diff_AB(self, func, x, n, m1, m2, eps=0.00001 + 0.00001j):
        """numerical center diff for comparison to autograd"""
        dz = []

        x1 = x.conj()
        fm = func(x1 - eps, n, m1, m2)
        fp = func(x1 + eps, n, m1, m2)
        dz.append((fp - fm) / (2 * eps))


        m11 = m1.conj()
        fm = func(x, n, m11 - eps, m2)
        fp = func(x, n, m11 + eps, m2)
        dz.append((fp - fm) / (2 * eps))

        m12 = m2.conj()
        fm = func(x1, n, m1, m12 - eps)
        fp = func(x1, n, m1, m12 + eps)
        dz.append((fp - fm) / (2 * eps))

        return dz




    def test_backwards_an(self):

        x = self.k * self.r_c
        y = self.k * self.r_s

        self.r_c.requires_grad = True
        self.r_s.requires_grad = True

        self.m1.requires_grad = True
        self.m2.requires_grad = True


        result_an = pmd.coreshell.an(x, y, self.n, self.m1, self.m2)
        dz_ad_an = torch.autograd.grad(
                    outputs=result_an,
                    inputs=[x, y, self.m1, self.m2],
                    grad_outputs=torch.ones_like(result_an),
                )

        dz_num_an = self.num_diff_ab(pmd.coreshell.an, x, y, self.n, self.m1, self.m2)
        print(dz_num_an)
        dz_num_an = torch.tensor(dz_num_an)

        torch.testing.assert_close(dz_ad_an[0], dz_num_an)













if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
