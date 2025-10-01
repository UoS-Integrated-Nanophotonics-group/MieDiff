import unittest

import torch
import numpy as np
import pymiediff as pmd
import random
import functools

from scipy.special import spherical_jn, spherical_yn


# hard coded equations using scipy as reference
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


def ab_sci(x, y, n, m1, m2):
    an = (
        psi(y, n) * (psi_der(m2 * y, n) - An_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * psi_der(y, n) * (psi(m2 * y, n) - An_sci(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        xi(y, n) * (psi_der(m2 * y, n) - An_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * xi_der(y, n) * (psi(m2 * y, n) - An_sci(x, n, m1, m2) * chi(m2 * y, n))
    )
    bn = (
        m2
        * psi(y, n)
        * (psi_der(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - psi_der(y, n) * (psi(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        m2 * xi(y, n) * (psi_der(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi_der(m2 * y, n))
        - xi_der(y, n) * (psi(m2 * y, n) - Bn_sci(x, n, m1, m2) * chi(m2 * y, n))
    )
    return an, bn


# ==============================================================================


# possible TODO - replace this with a test using Treams
class TestCoefficientsForwards(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        dtype_complex = torch.cdouble

        self.n_max = 5
        part_res = 10
        wl_res = 7

        self.n = torch.arange(self.n_max + 1)

        n_env = 1.0
        n_core = (
            (torch.linspace(1.5, 4, part_res) + 1j * torch.linspace(0, 1, part_res))
            .unsqueeze(-1)
            .broadcast_to(part_res, wl_res)
        )
        n_shell = (
            (torch.linspace(1.5, 4, part_res) + 1j * torch.linspace(0, 1, part_res))
            .unsqueeze(-1)
            .broadcast_to(part_res, wl_res)
        )

        self.x = (
            torch.linspace(0.5, 5, part_res)
            .unsqueeze(-1)
            .broadcast_to(part_res, wl_res)
        )

        self.y = self.x + torch.linspace(0.5, 5, part_res).unsqueeze(-1).broadcast_to(
            part_res, wl_res
        )
        self.m1 = n_core / n_env
        self.m2 = n_shell / n_env

    def tearDown(self):
        pass

    def test_forward_torch(self):
        kwargs_mie = {
            "x": self.x.to(torch.complex128),
            "y": self.y.to(torch.complex128),
            "m1": self.m1.to(torch.complex128),
            "m2": self.m2.to(torch.complex128),
        }

        configs = [
            ["scipy", "recurrence", 1e-5],
            ["torch", "recurrence", 1e-5],  # typically least accurate
            ["torch", "ratios", 1e-5],
        ]
        for conf in configs:
            backend = conf[0]
            which_jn = conf[1]
            tol = conf[2]

            if self.verbose:
                print("test vs scipy")

            result_ad = pmd.coreshell.ab(
                n=self.n_max, backend=backend, which_jn=which_jn, **kwargs_mie
            )

            kwargs_np = dict()
            for k in kwargs_mie:
                kwargs_np[k] = kwargs_mie[k].detach().cpu().numpy()
            result_scipy = torch.as_tensor(
                np.stack([ab_sci(n=_n, **kwargs_np) for _n in self.n[1:]], axis=(1))
            )

            torch.testing.assert_close(
                result_scipy[0], result_ad[0], rtol=tol, atol=tol
            )  # an
            torch.testing.assert_close(
                result_scipy[1], result_ad[1], rtol=tol, atol=tol
            )  # bn


if __name__ == "__main__":
    T = TestCoefficientsForwards()
    # T.setUp()
    # T.test_forward_scipy()
    # T.test_forward_torch()
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
