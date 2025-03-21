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
