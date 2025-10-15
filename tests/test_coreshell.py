import unittest
import warnings

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

            result_ad = pmd.coreshell._miecoef(
                n=self.n_max, backend=backend, which_jn=which_jn, **kwargs_mie
            )

            kwargs_np = dict()
            for k in kwargs_mie:
                kwargs_np[k] = kwargs_mie[k].detach().cpu().numpy()
            result_scipy = torch.as_tensor(
                np.stack([ab_sci(n=_n, **kwargs_np) for _n in self.n[1:]], axis=(1))
            )

            torch.testing.assert_close(
                result_scipy[0], result_ad["a_n"], rtol=tol, atol=tol
            )  # an
            torch.testing.assert_close(
                result_scipy[1], result_ad["b_n"], rtol=tol, atol=tol
            )  # bn


class TestCoefficientsForwardsTreams(unittest.TestCase):
    def test_mie_coeff_against_treams(self):
        # setup a core-shell test case

        # - config
        N_wl = 2
        wl0 = torch.linspace(500, 1500, N_wl)
        k0 = 2 * torch.pi / wl0

        r_core = 200.0
        r_shell = 420.0
        n_core = 0.5 + 5.1j
        n_shell = 4.5  # caution: metal-like shells can become unstable
        mat_core = pmd.materials.MatConstant(n_core**2)
        mat_shell = pmd.materials.MatConstant(n_shell**2)

        n_env = 1.0

        n_max = 10  # max Mie order

        # Mie from `treams`
        # https://github.com/tfp-photonics/treams
        try:
            import treams
        except ImportError:
            warnings.warn(
                "`treams` seems not installed. "
                + "Skipping Mie coefficient test vs `treams`."
            )
            return

        def mie_ab_sphere_treams(
            k0: np.ndarray, radii: list, materials: list, n_env, n_max
        ):
            """Mie scattering via package `treams`, for vacuum environment"""
            assert len(radii) == len(materials)

            # embedding medium: vacuum
            eps_env = n_env**2

            # main Mie extraction and setup
            a_n = np.zeros((len(k0), n_max), dtype=np.complex128)
            b_n = np.zeros_like(a_n)
            for i_wl, _k0 in enumerate(k0):

                # core and shell materials
                mat_treams = []
                for eps_mat in materials:
                    mat_treams.append(treams.Material(eps_mat))

                # treams convention: environment material last
                mat_treams.append(treams.Material(eps_env))

                for n in range(1, n_max + 1):
                    miecoef = treams.coeffs.mie(
                        n, _k0 * np.array(radii), *zip(*mat_treams)
                    )
                    a_n[i_wl, n - 1] = -miecoef[0, 0] - miecoef[0, 1]
                    b_n[i_wl, n - 1] = -miecoef[0, 0] + miecoef[0, 1]

            # scattering
            Qs_mie = np.zeros(len(k0))
            for n in range(n_max):
                Qs_mie += (2 * (n + 1) + 1) * (
                    np.abs(a_n[:, n]) ** 2 + np.abs(b_n[:, n]) ** 2
                )
            Qs_mie *= 2 / (k0 * n_env * max(radii)).real ** 2

            return dict(
                a_n=a_n,
                b_n=b_n,
                q_sca=Qs_mie,
                wavelength=2 * np.pi / k0,
            )

        cs_treams = mie_ab_sphere_treams(
            k0=k0.numpy(),
            radii=[r_core, r_shell],
            materials=[n_core**2, n_shell**2],
            n_env=n_env,
            n_max=n_max,
        )
        a_t = np.moveaxis(cs_treams["a_n"], 0, 1)
        b_t = np.moveaxis(cs_treams["b_n"], 0, 1)

        # pymiediff
        mie_ceof = pmd.coreshell.mie_coefficients(
            k0,
            r_c=r_core,
            eps_c=n_core**2,
            r_s=r_shell,
            eps_s=n_shell**2,
            eps_env=n_env**2,
            backend="torch",
            n_max=n_max,
        )

        a_pmd = mie_ceof["a_n"][:, 0]  # first particle
        b_pmd = mie_ceof["b_n"][:, 0]  # first particle

        a_p = a_pmd.cpu().detach().numpy()
        b_p = b_pmd.cpu().detach().numpy()

        # compare
        np.testing.assert_allclose(a_p, a_t, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(b_p, b_t, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    T = TestCoefficientsForwards()
    # T.setUp()
    # T.test_forward_scipy()
    # T.test_forward_torch()
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
