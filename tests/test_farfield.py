# encoding=utf-8
# %%
import unittest

import torch
import random
import numpy as np

import pymiediff as pmd
import PyMieScatt as pms
import treams

class TestFarFieldForward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

        # ======== Test Parameters ==========
        wl_res = 100
        N_pt_angular = 10

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

        self.wl0 = torch.linspace(starting_wavelength, ending_wavelength, wl_res)

        self.k0 = 2 * torch.pi / self.wl0

    def test_forward_efficiency(self):
        
        # ====== PyMieScatt test ========
        # Qsca_pms = []
        # Qext_pms = []
        # Qabs_pms = []

        # wl = self.wl0.numpy()

        # for wavelengh in wl:
        #     pms_results = pms.MieQCoreShell(
        #         mCore=self.core_refractiveIndex,
        #         mShell=self.shell_refractiveIndex,
        #         wavelength=wavelengh,
        #         dCore=2 * self.core_radius,
        #         dShell=2 * self.shell_radius,
        #         nMedium=1.0,
        #         asCrossSection=False,
        #         asDict=True,
        #     )

        #     Qsca_pms.append(pms_results["Qsca"])
        #     Qext_pms.append(pms_results["Qext"])
        #     Qabs_pms.append(pms_results["Qabs"])

        # pms_Qsca = torch.tensor(Qsca_pms, dtype=torch.double)
        # pms_Qabs = torch.tensor(Qabs_pms, dtype=torch.double)
        # pms_Qext = torch.tensor(Qext_pms, dtype=torch.double)

        # ====== Treams test =========
        n_max=10
        # embedding medium: vacuum
        eps_env = 1.0
        k0=self.k0.squeeze().numpy()
        radii=[self.core_radius, self.shell_radius], 
        materials=[self.core_refractiveIndex**2, self.shell_refractiveIndex**2]
        # main Mie extraction and setup
        a_n = np.zeros((len(k0), n_max), dtype=np.complex128)
        b_n = np.zeros_like(a_n)
        print(k0.shape)
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
            Qs_mie += (2 * (n + 1) + 1) * (np.abs(a_n[:, n]) ** 2 + np.abs(b_n[:, n]) ** 2)
            Qe_mie += (2 * (n + 1) + 1) * (np.real(a_n[:, n]) + np.real(b_n[:, n]))
        
        
        
        Qs_mie *= 2 / (k0 * max(radii)).real ** 2
        Qe_mie *= 2 / (k0 * max(radii)).real ** 2

        Qa_mie = Qe_mie - Qs_mie
        
        treams_Qsca = torch.tensor(Qs_mie, dtype=torch.double)
        treams_Qabs = torch.tensor(Qa_mie, dtype=torch.double)
        treams_Qext = torch.tensor(Qe_mie, dtype=torch.double)


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
        # CoreShellScatteringFunction
        i_per_pms = []
        i_par_pms = []
        i_unp_pms = []

        wl = self.wl0.numpy()

        for wavelengh in wl:
            theta, SL, SR, SU = pms.CoreShellScatteringFunction(
                mCore=self.core_refractiveIndex,
                mShell=self.shell_refractiveIndex,
                wavelength=wavelengh,
                dCore=2 * self.core_radius,
                dShell=2 * self.shell_radius,
                nMedium=1.0,
            )

            i_per_pms.append(SL)
            i_par_pms.append(SR)
            i_unp_pms.append(SU)

        theta = torch.tensor(theta, dtype=torch.double)

        pms_i_per = torch.tensor(np.array(i_per_pms), dtype=torch.double)
        pms_i_par = torch.tensor(np.array(i_par_pms), dtype=torch.double)
        pms_i_unp = torch.tensor(np.array(i_unp_pms), dtype=torch.double)

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
            pms_i_per, pmd_results["i_per"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            pms_i_par, pmd_results["i_par"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            pms_i_unp, pmd_results["i_unpol"], atol=1e-6, rtol=1e-6
        )


# class TestFarFieldBackward(unittest.TestCase):

#     def setUp(self):
#         self.verbose = False

#     def test_backwards(self):
#         self.assertTrue(False)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
