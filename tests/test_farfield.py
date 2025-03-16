# encoding=utf-8
# %%
import unittest

import torch
from scipy.special import spherical_jn, spherical_yn

import pymiediff as pmd


class TestFarFieldForward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

    def test_forward(self):
        self.assertTrue(False)


class TestFarFieldBackward(unittest.TestCase):

    def setUp(self):
        self.verbose = False

    def test_backwards(self):
        self.assertTrue(False)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
