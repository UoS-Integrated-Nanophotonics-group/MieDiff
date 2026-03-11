import unittest

import numpy as np
import torch

from scipy import special # spherical_jn, spherical_yn

import pymiediff as pmd


def scipy_psi(n, z):

    psi = []
    psi_der = []

    for k in n.numpy():

        jn = special.spherical_jn(k, z)

        jn_der = special.spherical_jn(k, z, derivative=True)

        psi.append(z * jn)
        psi_der.append(jn + z * jn_der)



    return np.stack(psi), np.stack(psi_der)



def scipy_xi(n, z):

    xi = []
    xi_der = []

    for k in n.numpy():

        jn = special.spherical_jn(k, z)
        yn = special.spherical_yn(k, z)

        jn_der = special.spherical_jn(k, z, derivative=True)
        yn_der = special.spherical_yn(k, z, derivative=True)

        xi = z * (jn + 1j * yn)
        xi_der = (jn + 1j * yn) + z * (jn_der + 1j * yn_der)

    return np.stack(xi), np.stack(xi_der)



class TestForward(unittest.TestCase):
    def test_D1n(self):
        # NOTE will fail for z = 0
        z_vals = np.linspace(0.01, 20, 50)  # 50 test points
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)

        n_max = 10
        n = torch.arange(0, n_max + 1)

        psi_scipy, psi_der_scipy = scipy_psi(n, z_vals)
        D1n_scipy = psi_der_scipy/psi_scipy

        # print(D1n_scipy[0])

        D1n_torch = pmd.special.D1n_torch(n_max, z_torch)
        D1n_torch_np = D1n_torch.detach().cpu().numpy()

        rel_err = np.max(np.abs((D1n_torch_np - D1n_scipy) / (D1n_scipy + 1e-16)))

        self.assertTrue(rel_err < 1e-7, f"D1n rel error too large: {rel_err}")



    def test_D3n_scipy(self):
        z_vals = np.linspace(0.01, 20, 50)  # 50 test points
        z_torch = torch.tensor(z_vals, dtype=torch.complex128)

        n_max = 10
        n = torch.arange(0, n_max + 1)

        xi_scipy, xi_der_scipy = scipy_xi(n, z_vals)
        D3n_scipy = xi_der_scipy/xi_scipy

        D1n_torch = pmd.special.D1n_torch(n_max, z_torch)
        D3n_torch, _ = pmd.special.D3n_torch(n_max, z_torch, D1n_torch)
        D3n_torch_np = D3n_torch.detach().cpu().numpy()

        rel_err = np.max(np.abs((D3n_torch_np - D3n_scipy) / (D3n_scipy + 1e-16)))

        self.assertTrue(rel_err < 1e-7, f"D3n rel error too large: {rel_err}")


    def test_Qln_scipy(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)