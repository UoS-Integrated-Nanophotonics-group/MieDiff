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
    return (2 * torch.pi / k**2) * (
        (2 * n1 + 1) * (a1.abs() ** 2 + b1.abs() ** 2)
        + (2 * n2 + 1) * (a2.abs() ** 2 + b2.abs() ** 2)
        + (2 * n3 + 1) * (a3.abs() ** 2 + b3.abs() ** 2)
        + (2 * n4 + 1) * (a4.abs() ** 2 + b4.abs() ** 2)
    )


def cross_ext(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4):
    return (2 * torch.pi / k**2) * (
        (2 * n1 + 1) * (a1 + b1).real
        + (2 * n2 + 1) * (a2 + b2).real
        + (2 * n3 + 1) * (a3 + b3).real
        + (2 * n4 + 1) * (a4 + b4).real
    )


def scs(k0, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
    if n_max is None:
        n_max = 10  # !! TODO: automatic eval. of adequate n_max.
    n = torch.arange(n_max).unsqueeze(0)  # dim. 0: spectral dimension (k0)
    assert len(n.shape) == 2

    # core-only: set shell == core
    if r_s is None:
        r_s = r_c
    if eps_s is None:
        eps_s = eps_c

    # convert everything to tensors
    k0 = torch.as_tensor(k0)
    k0 = torch.atleast_1d(k0)  # if single value, expand
    k0 = k0.unsqueeze(1)  # dim. 1: Mie order (n)
    assert len(k0.shape) == 2

    r_c = torch.as_tensor(r_c)
    r_s = torch.as_tensor(r_s)
    eps_c = torch.as_tensor(eps_c)
    eps_s = torch.as_tensor(eps_s)
    eps_env = torch.as_tensor(eps_env)

    n_c = torch.broadcast_to(torch.atleast_1d(eps_c).unsqueeze(1), k0.shape)**0.5
    n_s = torch.broadcast_to(torch.atleast_1d(eps_s).unsqueeze(1), k0.shape)**0.5
    n_env = torch.broadcast_to(torch.atleast_1d(eps_env).unsqueeze(1), k0.shape)**0.5

    # - eval Mie coefficients
    x = k0 * r_c
    y = k0 * r_s
    m_c = n_c / n_env
    m_s = n_s / n_env
    a_n = an(x, y, n, m_c, m_s)
    b_n = bn(x, y, n, m_c, m_s)

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 / (k0**2 * r_s**2)
    q_ext = torch.sum(prefactor * (2 * n + 1) * (a_n.real + b_n.real), dim=1)
    q_sca = torch.sum(
        prefactor
        * (2 * n + 1)
        * (a_n.real**2 + a_n.imag**2 + b_n.real**2 + b_n.imag**2),
        dim=1,
    )
    q_abs = q_ext - q_sca

    return dict(
        q_ext=q_ext,
        q_sca=q_sca,
        q_abs=q_abs,
        cs_geo=cs_geo,
        cs_ext=q_ext * cs_geo,
        cs_sca=q_sca * cs_geo,
        cs_abs=q_abs * cs_geo,
    )


# Does not passs gradcheck! no idea why. Error:
# While considering the real part of complex outputs only,
# Jacobian mismatch for output 0 with respect to input 3
# def cross_sca_new(k, r_c, r_s, m1, m2, n1, n2, n3, n4):
#     a1 = an(k * r_c, k * r_s, n1, m1, m2)
#     a2 = an(k * r_c, k * r_s, n2, m1, m2)
#     a3 = an(k * r_c, k * r_s, n3, m1, m2)
#     a4 = an(k * r_c, k * r_s, n4, m1, m2)

#     b1 = bn(k * r_c, k * r_s, n1, m1, m2)
#     b2 = bn(k * r_c, k * r_s, n2, m1, m2)
#     b3 = bn(k * r_c, k * r_s, n3, m1, m2)
#     b4 = bn(k * r_c, k * r_s, n4, m1, m2)

#     #print(a1.dtype)

#     return (2 * torch.pi / k**2) * (
#         (2 * n1 + 1) * (a1.abs() ** 2 + b1.abs() ** 2)
#         + (2 * n2 + 1) * (a2.abs() ** 2 + b2.abs() ** 2)
#         + (2 * n3 + 1) * (a3.abs() ** 2 + b3.abs() ** 2)
#         + (2 * n4 + 1) * (a4.abs() ** 2 + b4.abs() ** 2)
#     )

# This might not be too much of an issue as it may be better to let the
# scattering coeffients be inputs to the functions. This means they can be used
# in multiple ways (i.e. for sca and ext)
