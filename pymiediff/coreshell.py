import warnings

import numpy as np
import torch

# from . import special
from pymiediff import special  # use absolute package internal imports!


def An(x, n, m1, m2):
    return (
        m2 * special.psi(m2 * x, n) * special.psi_der(m1 * x, n)
        - m1 * special.psi_der(m2 * x, n) * special.psi(m1 * x, n)
    ) / (
        m2 * special.chi(m2 * x, n) * special.psi_der(m1 * x, n)
        - m1 * special.chi_der(m2 * x, n) * special.psi(m1 * x, n)
    )


def Bn(x, n, m1, m2):
    return (
        m2 * special.psi(m1 * x, n) * special.psi_der(m2 * x, n)
        - m1 * special.psi(m2 * x, n) * special.psi_der(m1 * x, n)
    ) / (
        m2 * special.chi_der(m2 * x, n) * special.psi(m1 * x, n)
        - m1 * special.psi_der(m1 * x, n) * special.chi(m2 * x, n)
    )


def an(x, y, n, m1, m2):
    return (
        special.psi(y, n)
        * (special.psi_der(m2 * y, n) - An(x, n, m1, m2) * special.chi_der(m2 * y, n))
        - m2
        * special.psi_der(y, n)
        * (special.psi(m2 * y, n) - An(x, n, m1, m2) * special.chi(m2 * y, n))
    ) / (
        special.xi(y, n)
        * (special.psi_der(m2 * y, n) - An(x, n, m1, m2) * special.chi_der(m2 * y, n))
        - m2
        * special.xi_der(y, n)
        * (special.psi(m2 * y, n) - An(x, n, m1, m2) * special.chi(m2 * y, n))
    )


def bn(x, y, n, m1, m2):
    return (
        m2
        * special.psi(y, n)
        * (special.psi_der(m2 * y, n) - Bn(x, n, m1, m2) * special.chi_der(m2 * y, n))
        - special.psi_der(y, n)
        * (special.psi(m2 * y, n) - Bn(x, n, m1, m2) * special.chi(m2 * y, n))
    ) / (
        m2
        * special.xi(y, n)
        * (special.psi_der(m2 * y, n) - Bn(x, n, m1, m2) * special.chi_der(m2 * y, n))
        - special.xi_der(y, n)
        * (special.psi(m2 * y, n) - Bn(x, n, m1, m2) * special.chi(m2 * y, n))
    )


def cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4):
    return ((2*torch.pi/k**2) *
            ((2*n1 + 1)*(a1.abs()**2 + b1.abs()**2) +
                (2*n2 + 1)*(a2.abs()**2 + b2.abs()**2) +
                (2*n3 + 1)*(a3.abs()**2 + b3.abs()**2) +
                (2*n4 + 1)*(a4.abs()**2 + b4.abs()**2)))