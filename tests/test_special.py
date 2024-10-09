# encoding=utf-8
# %%
import unittest

import torch
from scipy.special import spherical_jn, spherical_yn

import pymiediff as pmd


class TestSpecialFunctionsForward(unittest.TestCase):

    def setUp(self):
        # setup some random complex test arguments
        self.z = torch.rand(100, dtype=torch.complex128).unsqueeze(0)
        self.z -= 0.5 + 1j * 0.5
        self.z *= 50

        # test up to order 10
        self.n = torch.arange(0, 10).unsqueeze(1)

    def test_sph_jn(self):
        # eval autodiff implementation
        sph_jn_torch = pmd.special.Jn(self.n, self.z)

        # eval scipy implementation
        sph_jn_scipy = torch.as_tensor(
            spherical_jn(self.n.detach().cpu().numpy(), self.z.detach().cpu().numpy())
        )

        torch.testing.assert_close(sph_jn_scipy, sph_jn_torch)


class TestSpecialFunctionsBackward(unittest.TestCase):

    def setUp(self):
        # setup some random complex test arguments
        self.z = torch.rand(100, dtype=torch.complex128).unsqueeze(0)
        self.z -= 0.5 + 1j * 0.5
        self.z *= 50

    def num_dJn_dz(self, n, z, eps=0.0001 + 0.0001j):
        """numerical center diff for comparison to autograd"""
        z = z.conj()
        fm = pmd.special.Jn(n, z - eps)
        fp = pmd.special.Jn(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz

    def test_sph_jn(self):
        # test up to order 10
        for n in range(10):
            # autodiff
            self.z.requires_grad = True
            result = pmd.special.Jn(n, self.z)
            dz_ad = torch.autograd.grad(
                outputs=result,
                inputs=[self.z],
                grad_outputs=torch.ones_like(result),
            )

            # numerical center diff.
            dz_num = self.num_dJn_dz(n, self.z)

            torch.testing.assert_close(dz_ad[0], dz_num)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
