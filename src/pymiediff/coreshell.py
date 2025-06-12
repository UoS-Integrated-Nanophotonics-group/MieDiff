# -*- coding: utf-8 -*-
"""
Core-shell scattering coefficients

analytical solutions taken from

Bohren, Craig F., and Donald R. Huffman.
Absorption and scattering of light by small particles. John Wiley & Sons, 2008.
"""
import torch
from pymiediff.special import psi, psi_der, chi, chi_der, xi, xi_der
from pymiediff.special import Jn, Yn
from pymiediff.special import dJn, dYn
from pymiediff.special import sph_jn_torch, sph_yn_torch
from pymiediff.special import f_prime_torch


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


def ab(x, y, n, m1, m2, backend="scipy"):
    """an and bn scattering coefficients

    optimised to call the bessel functions the lowest amount of times

    Absorption and scattering of light by small particles.
    Pg. 183 and Equ. 8.1.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell
        backend (str, optional): backend to use for spherical bessel functions. Either 'scipy' or 'torch'. Defaults to 'scipy'.

    Returns:
        torch.Tensor: result
    """
    if backend.lower() == "scipy":
        return ab_scipy(x, y, n, m1, m2)
    elif backend.lower() in ["torch", "gpu"]:
        return ab_gpu(x, y, n, m1, m2)


def ab_scipy(x, y, n, m1, m2):
    """an and bn scattering coefficients

    optimised to call the bessel functions the lowest amount of times

    Absorption and scattering of light by small particles.
    Pg. 183 and Equ. 8.1.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
        torch.Tensor: result
    """
    # evaluate bessel terms
    j_y = Jn(n, y)
    y_y = Yn(n, y)
    j_m1x = Jn(n, m1 * x)
    j_m2x = Jn(n, m2 * x)
    j_m2y = Jn(n, m2 * y)

    y_m2x = Yn(n, m2 * x)
    y_m2y = Yn(n, m2 * y)

    h1_y = j_y + 1j * y_y

    # bessel derivatives
    dj_y = dJn(n, y)
    dj_m1x = dJn(n, m1 * x)
    dj_m2x = dJn(n, m2 * x)
    dj_m2y = dJn(n, m2 * y)
    dy_y = dYn(n, y)
    dy_m2x = dYn(n, m2 * x)
    dy_m2y = dYn(n, m2 * y)

    dh1_y = dj_y + 1j * dy_y

    # eval. psi, chi, xi terms
    psi_y = y * j_y
    psi_m1x = (m1 * x) * j_m1x
    psi_m2x = (m2 * x) * j_m2x
    psi_m2y = (m2 * y) * j_m2y

    chi_m2x = -(m2 * x) * y_m2x
    chi_m2y = -(m2 * y) * y_m2y

    xi_y = y * h1_y

    dpsi_y = j_y + y * dj_y
    dpsi_m1x = j_m1x + (m1 * x) * dj_m1x
    dpsi_m2x = j_m2x + (m2 * x) * dj_m2x
    dpsi_m2y = j_m2y + (m2 * y) * dj_m2y

    dchi_m2x = -y_m2x - (m2 * x) * dy_m2x
    dchi_m2y = -y_m2y - (m2 * y) * dy_m2y

    dxi_y = h1_y + y * dh1_y

    # Mie coeffs.
    An = (m2 * psi_m2x * dpsi_m1x - m1 * dpsi_m2x * psi_m1x) / (
        m2 * chi_m2x * dpsi_m1x - m1 * dchi_m2x * psi_m1x
    )
    Bn = (m2 * psi_m1x * dpsi_m2x - m1 * psi_m2x * dpsi_m1x) / (
        m2 * dchi_m2x * psi_m1x - m1 * dpsi_m1x * chi_m2x
    )

    an = (
        psi_y * (dpsi_m2y - An * dchi_m2y) - m2 * dpsi_y * (psi_m2y - An * chi_m2y)
    ) / (xi_y * (dpsi_m2y - An * dchi_m2y) - m2 * dxi_y * (psi_m2y - An * chi_m2y))
    bn = (
        m2 * psi_y * (dpsi_m2y - Bn * dchi_m2y) - dpsi_y * (psi_m2y - Bn * chi_m2y)
    ) / (m2 * xi_y * (dpsi_m2y - Bn * dchi_m2y) - dxi_y * (psi_m2y - Bn * chi_m2y))
    return an, bn


def ab_gpu(x, y, n, m1, m2):
    """an and bn scattering coefficients - GPU compatible implementation

    native torch implementation via bessel upward and downward recurrences

    Absorption and scattering of light by small particles.
    Pg. 183 and Equ. 8.1.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
        torch.Tensor: result
    """
    # use recurrence implementations
    # !! caution with vectorization !!
    # Fix broadcasting!!

    # n = torch.max(torch.as_tensor(n))

    # evaluate bessel terms
    j_y = sph_jn_torch(n, y)
    y_y = sph_yn_torch(n, y)
    j_m1x = sph_jn_torch(n, m1 * x)
    j_m2x = sph_jn_torch(n, m2 * x)
    j_m2y = sph_jn_torch(n, m2 * y)

    y_m2x = sph_yn_torch(n, m2 * x)
    y_m2y = sph_yn_torch(n, m2 * y)

    h1_y = j_y + 1j * y_y

    # bessel derivatives
    dj_y = f_prime_torch(n, y, j_y)
    dj_m1x = f_prime_torch(n, m1 * x, j_m1x)
    dj_m2x = f_prime_torch(n, m2 * x, j_m2x)
    dj_m2y = f_prime_torch(n, m2 * y, j_m2y)
    dy_y = f_prime_torch(n, y, y_y)
    dy_m2x = f_prime_torch(n, m2 * x, y_m2x)
    dy_m2y = f_prime_torch(n, m2 * y, y_m2y)

    dh1_y = dj_y + 1j * dy_y

    # use only required Mie orders
    j_y = j_y
    y_y = y_y
    j_m1x = j_m1x
    j_m2x = j_m2x
    j_m2y = j_m2y
    y_m2x = y_m2x
    y_m2y = y_m2y
    h1_y = h1_y

    # eval. psi, chi, xi terms
    psi_y = y * j_y
    psi_m1x = (m1 * x) * j_m1x
    psi_m2x = (m2 * x) * j_m2x
    psi_m2y = (m2 * y) * j_m2y

    chi_m2x = -(m2 * x) * y_m2x
    chi_m2y = -(m2 * y) * y_m2y

    xi_y = y * h1_y

    dpsi_y = j_y + y * dj_y
    dpsi_m1x = j_m1x + (m1 * x) * dj_m1x
    dpsi_m2x = j_m2x + (m2 * x) * dj_m2x
    dpsi_m2y = j_m2y + (m2 * y) * dj_m2y

    dchi_m2x = -y_m2x - (m2 * x) * dy_m2x
    dchi_m2y = -y_m2y - (m2 * y) * dy_m2y

    dxi_y = h1_y + y * dh1_y

    # Mie coeffs.
    m1_bc = m1
    m2_bc = m2
    An = (m2_bc * psi_m2x * dpsi_m1x - m1_bc * dpsi_m2x * psi_m1x) / (
        m2_bc * chi_m2x * dpsi_m1x - m1_bc * dchi_m2x * psi_m1x
    )
    Bn = (m2_bc * psi_m1x * dpsi_m2x - m1_bc * psi_m2x * dpsi_m1x) / (
        m2_bc * dchi_m2x * psi_m1x - m1_bc * dpsi_m1x * chi_m2x
    )

    an = (
        psi_y * (dpsi_m2y - An * dchi_m2y) - m2_bc * dpsi_y * (psi_m2y - An * chi_m2y)
    ) / (xi_y * (dpsi_m2y - An * dchi_m2y) - m2_bc * dxi_y * (psi_m2y - An * chi_m2y))
    bn = (
        m2_bc * psi_y * (dpsi_m2y - Bn * dchi_m2y) - dpsi_y * (psi_m2y - Bn * chi_m2y)
    ) / (m2_bc * xi_y * (dpsi_m2y - Bn * dchi_m2y) - dxi_y * (psi_m2y - Bn * chi_m2y))

    # 0/0
    # recurrences return n+1 orders: remove last order
    return an[..., 1:], bn[..., 1:]


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
    # TODO - optimize runtime
    # combine c_n and d_n and evaluate each required term only once

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
    # TODO - optimize runtime
    # combine c_n and d_n and evaluate each required term only once

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
