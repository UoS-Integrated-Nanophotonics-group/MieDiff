# encoding=utf-8
# %%
import unittest

import torch
from scipy.special import spherical_jn, spherical_yn

import pymiediff as pmd


class TestSpecialFunctionsForward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

        # setup some random complex test arguments
        self.z = torch.rand(100, dtype=torch.complex128).unsqueeze(0)
        self.z -= 0.5 + 1j * 0.5
        self.z *= 50

        # test up to order 10
        self.n = torch.arange(0, 10).unsqueeze(1)

    def test_forward(self):
        function_sets = [
            (pmd.special.Jn, spherical_jn, {}),
            (pmd.special.Yn, spherical_yn, {}),
            (pmd.special.dJn, spherical_jn, {"derivative": True}),
            (pmd.special.dYn, spherical_yn, {"derivative": True}),
        ]

        for func_ad, func_scipy, kwargs in function_sets:
            if self.verbose:
                print("test vs scipy: ", func_ad)

            # eval autodiff implementation
            sph_jn_torch = func_ad(self.n, self.z)

            # eval scipy implementation
            sph_jn_scipy = torch.as_tensor(
                func_scipy(
                    self.n.detach().cpu().numpy(),
                    self.z.detach().cpu().numpy(),
                    **kwargs
                )
            )

            torch.testing.assert_close(sph_jn_scipy, sph_jn_torch)


class TestSpecialFunctionsBackward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

        # setup some random complex test arguments
        self.z = torch.rand(100, dtype=torch.complex128).unsqueeze(0)
        self.z -= 0.5 + 1j * 0.5
        self.z *= 50

    def num_diff(self, func, n, z, eps=0.00001 + 0.00001j):
        """numerical center diff for comparison to autograd"""
        z = z.conj()
        fm = func(n, z - eps)
        fp = func(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz

    def test_backwards(self):
        function_sets = [
            pmd.special.Jn,
            pmd.special.Yn,
            pmd.special.dJn,
            pmd.special.dYn,
        ]

        for func_ad in function_sets:
            if self.verbose:
                print("test autodiff vs num. diff.: ", func_ad)

            # test up to order 10
            for n in range(10):
                # autodiff
                self.z.requires_grad = True
                result = func_ad(n, self.z)
                dz_ad = torch.autograd.grad(
                    outputs=result,
                    inputs=[self.z],
                    grad_outputs=torch.ones_like(result),
                )

                # numerical center diff.
                dz_num = self.num_diff(func_ad, n, self.z)

                torch.testing.assert_close(dz_ad[0], dz_num)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
