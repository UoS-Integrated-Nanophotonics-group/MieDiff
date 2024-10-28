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


class TestCoefficientsForwards(unittest.TestCase):

    def setUp(self):
        self.verbose = True

        # wlRes = 1000
        # z = 2 * np.pi / np.linspace(200, 600, wlRes)

        self.x = torch.rand(100, dtype=torch.complex128).unsqueeze(0)
        self.x -= 0.5 + 1j * 0.5
        self.x *= 50
        self.y = torch.rand(100, dtype=torch.complex128).unsqueeze(0)
        self.y -= 0.5 + 1j * 0.5
        self.y *= 50

        self.m1 = torch.tensor(2.0)  # 1*np.random.normal() + 1j*np.random.normal())
        self.m2 = torch.tensor(5.0)  # 1*np.random.normal() + 1j*np.random.normal())

        self.n = torch.tensor(5.0)

    def test_forward(self):
        function_sets = [
            (pmd.an, an_sci, {}),
            (pmd.bn, bn_sci, {}),
        ]

        for func_ad, func_scipy, kwargs in function_sets:
            if self.verbose:
                print("test vs scipy: ", func_ad)

            result_ad = func_ad(
                self.x,
                self.y,
                self.n,
                self.m1,
                self.m2,
            )

            result_scipy = torch.as_tensor(
                func_scipy(
                    self.x.detach().cpu().numpy(),
                    self.y.detach().cpu().numpy(),
                    self.n.detach().cpu().numpy(),
                    self.m1.detach().cpu().numpy(),
                    self.m2.detach().cpu().numpy(),
                    **kwargs
                )
            )

            torch.testing.assert_close(result_scipy, result_ad)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
