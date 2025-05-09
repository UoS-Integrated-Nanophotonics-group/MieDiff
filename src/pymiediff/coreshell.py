# -*- coding: utf-8 -*-
"""
Core-shell scattering coefficients

analytical solutions taken from

Bohren, Craig F., and Donald R. Huffman.
Absorption and scattering of light by small particles. John Wiley & Sons, 2008.
"""
from pymiediff.special import psi, psi_der, chi, chi_der, xi, xi_der


def _An(x, n, m1, m2):
    """private An scattering coefficient

    Absorption and scattering of light by small particles.
    Pg. and Equ. number.

    Args:
        x (torch.Tensor): size parameter (core)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
        torch.Tensor: result
    """
    return (
        m2 * psi(m2 * x, n) * psi_der(m1 * x, n)
        - m1 * psi_der(m2 * x, n) * psi(m1 * x, n)
    ) / (
        m2 * chi(m2 * x, n) * psi_der(m1 * x, n)
        - m1 * chi_der(m2 * x, n) * psi(m1 * x, n)
    )


def _Bn(x, n, m1, m2):
    """private Bn scattering coefficient

    Absorption and scattering of light by small particles.
    Pg. and Equ. number.

    Args:
        x (torch.Tensor): size parameter (core)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
        torch.Tensor: result
    """
    return (
        m2 * psi(m1 * x, n) * psi_der(m2 * x, n)
        - m1 * psi(m2 * x, n) * psi_der(m1 * x, n)
    ) / (
        m2 * chi_der(m2 * x, n) * psi(m1 * x, n)
        - m1 * psi_der(m1 * x, n) * chi(m2 * x, n)
    )


def ab(x, y, n, m1, m2):
    """an scattering coefficient

    Absorption and scattering of light by small particles.
    Pg. and Equ. number.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
        torch.Tensor: result
    """
    psi_yn = psi(y, n)
    psi_m2yn = psi(m2 * y, n)
    psi_m1xn = psi(m1 * x, n)
    psi_m2xn = psi(m2 * x, n)
    chi_m2yn = chi(m2 * y, n)
    chi_m2xn = chi(m2 * x, n)
    dpsi_m2yn = psi_der(m2 * y, n)
    dpsi_m1xn = psi_der(m1 * x, n)
    dpsi_m2xn = psi_der(m2 * x, n)
    dpsi_yn = psi_der(y, n)
    dchi_m2yn = chi_der(m2 * y, n)
    dchi_m2xn = chi_der(m2 * x, n)
    dxi_yn = xi_der(y, n)


    An = (m2 * psi_m2xn * dpsi_m1xn - m1 * dpsi_m2xn * psi_m1xn) / (
        m2 * chi_m2xn * dpsi_m1xn - m1 * dchi_m2xn * psi_m1xn
    )
    Bn = (m2 * psi_m1xn * dpsi_m2xn - m1 * psi_m2xn * dpsi_m1xn) / (
        m2 * dchi_m2xn * psi_m1xn - m1 * dpsi_m1xn * chi_m2xn
    )

    an = (
        psi_yn * (dpsi_m2yn - An * dchi_m2yn)
        - m2 * dpsi_yn * (psi_m2yn - An * chi_m2yn)
    ) / (
        xi(y, n) * (dpsi_m2yn - An * dchi_m2yn)
        - m2 * dxi_yn * (psi_m2yn - An * chi_m2yn)
    )
    bn = (
        m2 * psi_yn * (dpsi_m2yn - Bn * dchi_m2yn)
        - dpsi_yn * (psi_m2yn - Bn * chi_m2yn)
    ) / (
        m2 * xi(y, n) * (dpsi_m2yn - Bn * dchi_m2yn)
        - dxi_yn * (psi_m2yn - Bn * chi_m2yn)
    )
    return an, bn


def an(x, y, n, m1, m2):
    """an scattering coefficient

    Absorption and scattering of light by small particles.
    Pg. and Equ. number.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
        torch.Tensor: result
    """
    return (
        psi(y, n) * (psi_der(m2 * y, n) - _An(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * psi_der(y, n) * (psi(m2 * y, n) - _An(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        xi(y, n) * (psi_der(m2 * y, n) - _An(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * xi_der(y, n) * (psi(m2 * y, n) - _An(x, n, m1, m2) * chi(m2 * y, n))
    )


def bn(x, y, n, m1, m2):
    """bn scattering coefficient

    Absorption and scattering of light by small particles.
    Pg. and Equ. number.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
       torch.Tensor: result
    """
    return (
        m2 * psi(y, n) * (psi_der(m2 * y, n) - _Bn(x, n, m1, m2) * chi_der(m2 * y, n))
        - psi_der(y, n) * (psi(m2 * y, n) - _Bn(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        m2 * xi(y, n) * (psi_der(m2 * y, n) - _Bn(x, n, m1, m2) * chi_der(m2 * y, n))
        - xi_der(y, n) * (psi(m2 * y, n) - _Bn(x, n, m1, m2) * chi(m2 * y, n))
    )


def cn(x, y, n, m1, m2):
    """cn scattering coefficient

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
       torch.Tensor: result
    """
    return (
        chi_der(m2 * x, n) * m1 * m2 * psi_der(y, n) * psi(m2 * x, n) * xi(y, n)
        - chi_der(m2 * x, n) * m1 * m2 * psi(m2 * x, n) * psi(y, n) * xi_der(y, n)
        - chi(m2 * x, n) * m1 * m2 * psi_der(m2 * x, n) * psi_der(y, n) * xi(y, n)
        + chi(m2 * x, n) * m1 * m2 * psi_der(m2 * x, n) * psi(y, n) * xi_der(y, n)
    ) / (
        chi_der(m2 * x, n) * m2**2 * psi_der(m2 * y, n) * psi(m1 * x, n) * xi(y, n)
        - chi_der(m2 * x, n) * m2 * psi(m1 * x, n) * psi(m2 * y, n) * xi_der(y, n)
        + chi_der(m2 * y, n) * m1 * m2 * psi_der(m1 * x, n) * psi(m2 * x, n) * xi(y, n)
        - chi_der(m2 * y, n) * m2**2 * psi_der(m2 * x, n) * psi(m1 * x, n) * xi(y, n)
        - chi(m2 * x, n) * m1 * m2 * psi_der(m1 * x, n) * psi_der(m2 * y, n) * xi(y, n)
        + chi(m2 * x, n) * m1 * psi_der(m1 * x, n) * psi(m2 * y, n) * xi_der(y, n)
        - chi(m2 * y, n) * m1 * psi_der(m1 * x, n) * psi(m2 * x, n) * xi_der(y, n)
        + chi(m2 * y, n) * m2 * psi_der(m2 * x, n) * psi(m1 * x, n) * xi_der(y, n)
    )


def dn(x, y, n, m1, m2):
    """dn scattering coefficient

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
       torch.Tensor: result
    """
    return (
        -chi_der(m2 * x, n) * m1 * m2 * psi_der(y, n) * psi(m2 * x, n) * xi(y, n)
        + chi_der(m2 * x, n) * m1 * m2 * psi(m2 * x, n) * psi(y, n) * xi_der(y, n)
        + chi(m2 * x, n) * m1 * m2 * psi_der(m2 * x, n) * psi_der(y, n) * xi(y, n)
        - chi(m2 * x, n) * m1 * m2 * psi_der(m2 * x, n) * psi(y, n) * xi_der(y, n)
    ) / (
        chi_der(m2 * x, n) * m1 * m2 * psi(m1 * x, n) * psi(m2 * y, n) * xi_der(y, n)
        - chi_der(m2 * x, n) * m1 * psi_der(m2 * y, n) * psi(m1 * x, n) * xi(y, n)
        + chi_der(m2 * y, n) * m1 * psi_der(m2 * x, n) * psi(m1 * x, n) * xi(y, n)
        - chi_der(m2 * y, n) * m2 * psi_der(m1 * x, n) * psi(m2 * x, n) * xi(y, n)
        - chi(m2 * x, n) * m2**2 * psi_der(m1 * x, n) * psi(m2 * y, n) * xi_der(y, n)
        + chi(m2 * x, n) * m2 * psi_der(m1 * x, n) * psi_der(m2 * y, n) * xi(y, n)
        - chi(m2 * y, n) * m1 * m2 * psi_der(m2 * x, n) * psi(m1 * x, n) * xi_der(y, n)
        + chi(m2 * y, n) * m2**2 * psi_der(m1 * x, n) * psi(m2 * x, n) * xi_der(y, n)
    )
