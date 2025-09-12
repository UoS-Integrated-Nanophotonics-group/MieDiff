# encoding=utf-8
# %%
import unittest

import torch
import random
import numpy as np
import treams.special

import pymiediff as pmd
import treams


class TestFarFieldForward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

        # ======== Test Parameters ==========
        wl_res = 100
        angular_res = 10

        # Test Case for testing:
        self.core_radius = random.uniform(1, 20)  # nm
        self.shell_radius = self.core_radius + random.uniform(1, 20)  # nm
        self.core_refractiveIndex = random.uniform(0.1, 4.0) + 1j * random.uniform(
            0.01, 1.0
        )
        self.shell_refractiveIndex = random.uniform(0.1, 4.0) + 1j * random.uniform(
            0.01, 1.0
        )
        starting_wavelength = 100.0  # nm
        ending_wavelength = 2000.0  # nm
        # ===================================

        self.theta = np.linspace(0.0001, 2 * np.pi + 0.0001, angular_res)

        self.wl0 = torch.linspace(starting_wavelength, ending_wavelength, wl_res)

        self.k0 = 2 * torch.pi / self.wl0

    def test_forward_efficiency(self):

        # ====== Treams test =========
        n_max = 10
        # embedding medium: vacuum
        eps_env = 1.0
        k0 = self.k0.numpy()
        radii = [self.core_radius, self.shell_radius]
        materials = [self.core_refractiveIndex**2, self.shell_refractiveIndex**2]
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
                miecoef = treams.coeffs.mie(n, _k0 * np.array(radii), *zip(*mat_treams))
                a_n[i_wl, n - 1] = -miecoef[0, 0] - miecoef[0, 1]
                b_n[i_wl, n - 1] = -miecoef[0, 0] + miecoef[0, 1]

        # scattering
        Qs_mie = np.zeros(len(k0))
        Qe_mie = np.zeros(len(k0))
        Qa_mie = np.zeros(len(k0))
        for n in range(n_max):
            Qs_mie += (2 * (n + 1) + 1) * (
                np.abs(a_n[:, n]) ** 2 + np.abs(b_n[:, n]) ** 2
            )
            Qe_mie += (2 * (n + 1) + 1) * (np.real(a_n[:, n]) + np.real(b_n[:, n]))

        Qs_mie *= 2 / (k0 * max(radii)).real ** 2
        Qe_mie *= 2 / (k0 * max(radii)).real ** 2

        Qa_mie = Qe_mie - Qs_mie

        treams_Qsca = torch.tensor(Qs_mie, dtype=torch.double).unsqueeze(0)
        treams_Qabs = torch.tensor(Qa_mie, dtype=torch.double).unsqueeze(0)
        treams_Qext = torch.tensor(Qe_mie, dtype=torch.double).unsqueeze(0)

        r_c = torch.tensor(self.core_radius, dtype=torch.double)
        r_s = torch.tensor(self.shell_radius, dtype=torch.double)
        n_c = torch.tensor(self.core_refractiveIndex, dtype=torch.cdouble)
        n_s = torch.tensor(self.shell_refractiveIndex, dtype=torch.cdouble)

        pmd_results = pmd.farfield.cross_sections(
            k0=self.k0,
            r_c=r_c,
            eps_c=n_c**2,
            r_s=r_s,
            eps_s=n_s**2,
            eps_env=1,
        )

        torch.testing.assert_close(
            treams_Qsca, pmd_results["q_sca"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            treams_Qabs, pmd_results["q_abs"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            treams_Qext, pmd_results["q_ext"], atol=1e-6, rtol=1e-6
        )

    def test_forward_angular(self):

        theta = torch.tensor(self.theta, dtype=torch.double)

        # ====== Treams test =========
        n_max = 10
        # embedding medium: vacuum
        eps_env = 1.0
        k0 = self.k0.numpy()
        radii = [self.core_radius, self.shell_radius]
        materials = [self.core_refractiveIndex**2, self.shell_refractiveIndex**2]
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
                miecoef = treams.coeffs.mie(n, _k0 * np.array(radii), *zip(*mat_treams))
                a_n[i_wl, n - 1] = -miecoef[0, 0] - miecoef[0, 1]
                b_n[i_wl, n - 1] = -miecoef[0, 0] + miecoef[0, 1]

        pi_n = np.zeros((len(k0), n_max, len(theta)), dtype=np.complex128)
        tau_n = np.zeros((len(k0), n_max, len(theta)), dtype=np.complex128)

        for i_theta, _theta in enumerate(theta):
            for i_wl, _k0 in enumerate(k0):
                for n in range(1, n_max + 1):
                    # print(i_theta, i_wl, n)
                    # https://tfp-photonics.github.io/treams/generated/treams.special.pi_fun.html#treams.special.pi_fun
                    pi_n[i_wl, i_theta, n - 1] = treams.special.pi_fun(
                        n, 1, np.cos(_theta)
                    )  # /np.sqrt(1-np.cos(_theta)**2)

                    # https://tfp-photonics.github.io/treams/generated/treams.special.tau_fun.html#treams.special.tau_fun
                    tau_n[i_wl, i_theta, n - 1] = treams.special.tau_fun(
                        n, 1, np.cos(_theta)
                    )

        s1_treams = np.zeros((len(k0), len(theta)), dtype=np.complex128)
        s2_treams = np.zeros((len(k0), len(theta)), dtype=np.complex128)

        for n in range(1, n_max + 1):
            s1_treams += ((2 * n + 1) / (n * (n + 1))) * (
                a_n[:, n - 1].reshape(-1, 1) * pi_n[:, :, n - 1]
                + b_n[:, n - 1].reshape(-1, 1) * tau_n[:, :, n - 1]
            )
            s2_treams += ((2 * n + 1) / (n * (n + 1))) * (
                a_n[:, n - 1].reshape(-1, 1) * tau_n[:, :, n - 1]
                + b_n[:, n - 1].reshape(-1, 1) * pi_n[:, :, n - 1]
            )

        treams_i_per = torch.tensor(np.abs(s1_treams) ** 2).unsqueeze(0)
        treams_i_par = torch.tensor(np.abs(s2_treams) ** 2).unsqueeze(0)

        treams_i_unp = (treams_i_par + treams_i_per) / 2

        r_c = torch.tensor(self.core_radius, dtype=torch.double)
        r_s = torch.tensor(self.shell_radius, dtype=torch.double)
        n_c = torch.tensor(self.core_refractiveIndex, dtype=torch.cdouble)
        n_s = torch.tensor(self.shell_refractiveIndex, dtype=torch.cdouble)

        pmd_results = pmd.farfield.angular_scattering(
            k0=self.k0,
            theta=theta,
            r_c=r_c,
            eps_c=n_c**2,
            r_s=r_s,
            eps_s=n_s**2,
            eps_env=1,
        )

        torch.testing.assert_close(
            treams_i_per, pmd_results["i_per"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            treams_i_par, pmd_results["i_par"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            treams_i_unp, pmd_results["i_unpol"], atol=1e-6, rtol=1e-6
        )


# Possible TODO
# class TestFarFieldBackward(unittest.TestCase):

#     def setUp(self):
#         self.verbose = False

#     def test_backwards(self):
#         self.assertTrue(False)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
