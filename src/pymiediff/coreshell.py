# -*- coding: utf-8 -*-
"""
Core-shell scattering coefficients

analytical solutions taken from

Bohren, Craig F., and Donald R. Huffman.
Absorption and scattering of light by small particles. John Wiley & Sons, 2008.
"""
import warnings

import torch

from pymiediff import special
from pymiediff import helper


# - Mie coefficients
def ab(x, y, n, m1, m2, backend="torch", precision="double", which_jn="recurrence"):
    """an and bn scattering coefficients

    native torch implementation via bessel upward and downward recurrences

    Absorption and scattering of light by small particles.
    Pg. 183 and Equ. 8.1.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell
        backend (str): Which backend to use. "scipy" or "torch". Defaults to "torch".
        precision (str, optional): "single" our "double". defaults to "double".
        which_jn (str): Which algorithm to use for spherical Bessel of first kind (j_n).
            must be one of: ['recurrence', 'ratios'], indicating, respectively, continued fractions ratios
            or simple downward recurrence. Defaults to 'recurrence'

    Returns:
        torch.Tensor, torch.Tensor: result
    """
    # backend selection
    if backend.lower() in ["torch"] and which_jn.lower() != "recurrence":
        sph_jn_func = special.sph_jn_torch
        sph_yn_func = special.sph_yn_torch
    elif backend.lower() in ["torch"] and which_jn.lower() == "recurrence":
        sph_jn_func = special.sph_jn_torch_via_rec
        sph_yn_func = special.sph_yn_torch
    elif backend.lower() in ["scipy"]:
        sph_jn_func = special.sph_jn
        sph_yn_func = special.sph_yn
    else:
        raise ValueError("Unknown backend configuration.")

    # evaluate bessel terms
    j_y = sph_jn_func(n, y, precision=precision)
    y_y = sph_yn_func(n, y, precision=precision)
    j_m1x = sph_jn_func(n, m1 * x, precision=precision)
    j_m2x = sph_jn_func(n, m2 * x, precision=precision)
    j_m2y = sph_jn_func(n, m2 * y, precision=precision)

    y_m2x = sph_yn_func(n, m2 * x, precision=precision)
    y_m2y = sph_yn_func(n, m2 * y, precision=precision)

    h1_y = j_y + 1j * y_y

    # bessel derivatives
    dj_y = special.f_der(n, y, j_y, precision=precision)
    dj_m1x = special.f_der(n, m1 * x, j_m1x, precision=precision)
    dj_m2x = special.f_der(n, m2 * x, j_m2x, precision=precision)
    dj_m2y = special.f_der(n, m2 * y, j_m2y, precision=precision)
    dy_y = special.f_der(n, y, y_y, precision=precision)
    dy_m2x = special.f_der(n, m2 * x, y_m2x, precision=precision)
    dy_m2y = special.f_der(n, m2 * y, y_m2y, precision=precision)

    dh1_y = dj_y + 1j * dy_y

    # eval. ricatti-bessel terms (psi, chi, xi)
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

    # recurrences return n+1 orders: remove zeroth order
    return an[1:, ...], bn[1:, ...]


def cd(x, y, n, m1, m2):
    """internal Mie coefficients c_n, d_n for the core

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell

    Returns:
       torch.Tensor, torch.Tensor: result
    """
    # Precompute all needed terms once
    psi_y, psi_y_der = special.psi(y, n)
    xi_y, xi_y_der = special.xi(y, n)
    psi_m1x, psi_m1x_der = special.psi(m1 * x, n)
    psi_m2x, psi_m2x_der = special.psi(m2 * x, n)
    psi_m2y, psi_m2y_der = special.psi(m2 * y, n)
    chi_m2x, chi_m2x_der = special.chi(m2 * x, n)
    chi_m2y, chi_m2y_der = special.chi(m2 * y, n)

    # cn expression
    cn = (
        chi_m2x_der * m1 * m2 * psi_y_der * psi_m2x * xi_y
        - chi_m2x_der * m1 * m2 * psi_m2x * psi_y * xi_y_der
        - chi_m2x * m1 * m2 * psi_m2x_der * psi_y_der * xi_y
        + chi_m2x * m1 * m2 * psi_m2x_der * psi_y * xi_y_der
    ) / (
        chi_m2x_der * m2**2 * psi_m2y_der * psi_m1x * xi_y
        - chi_m2x_der * m2 * psi_m1x * psi_m2y * xi_y_der
        + chi_m2y_der * m1 * m2 * psi_m1x_der * psi_m2x * xi_y
        - chi_m2y_der * m2**2 * psi_m2x_der * psi_m1x * xi_y
        - chi_m2x * m1 * m2 * psi_m1x_der * psi_m2y_der * xi_y
        + chi_m2x * m1 * psi_m1x_der * psi_m2y * xi_y_der
        - chi_m2y * m1 * psi_m1x_der * psi_m2x * xi_y_der
        + chi_m2y * m2 * psi_m2x_der * psi_m1x * xi_y_der
    )

    # dn expression
    dn = (
        -chi_m2x_der * m1 * m2 * psi_y_der * psi_m2x * xi_y
        + chi_m2x_der * m1 * m2 * psi_m2x * psi_y * xi_y_der
        + chi_m2x * m1 * m2 * psi_m2x_der * psi_y_der * xi_y
        - chi_m2x * m1 * m2 * psi_m2x_der * psi_y * xi_y_der
    ) / (
        chi_m2x_der * m1 * m2 * psi_m1x * psi_m2y * xi_y_der
        - chi_m2x_der * m1 * psi_m2y_der * psi_m1x * xi_y
        + chi_m2y_der * m1 * psi_m2x_der * psi_m1x * xi_y
        - chi_m2y_der * m2 * psi_m1x_der * psi_m2x * xi_y
        - chi_m2x * m2**2 * psi_m1x_der * psi_m2y * xi_y_der
        + chi_m2x * m2 * psi_m1x_der * psi_m2y_der * xi_y
        - chi_m2y * m1 * m2 * psi_m2x_der * psi_m1x * xi_y_der
        + chi_m2y * m2**2 * psi_m1x_der * psi_m2x * xi_y_der
    )
    return cn, dn


## TODO: Test c,d and implement also f, g, v, w coefficients (inside shell)
# (see Bohren & Huffmann, chap. 8.1)


# - internal helper
def _broadcast_mie_config(k0, r_c, r_s, eps_c, eps_s, eps_env):
    """broadcast configs to dimension for vectorization

    dimension convention:
    (n Mie order, N particles, N wavevectors)

    Here, expand all parameters to match the convention (N_part, N_k0)

    Args:
        k0 (tensor of float): wavevector, shape (N k0)
        r_c (tensor of float): core radius, shape (N particles)
        r_s (tensor of float): shell radius, shape (N particles)
        eps_c (tensor of complex): core permittivity, shape (N particles, N k0)
        eps_s (tensor of complex): shell permittivity, shape (N particles, N k0)
        eps_env (tensor of float): environemnt permittivity, shape (N k0)

    Returns:
        same as input, but all cast to dim 3
    """
    # convert everything to tensors
    k0 = torch.as_tensor(k0)
    k0 = k0.squeeze()  # remove possible empty dimensions
    k0 = torch.atleast_1d(k0)  # if single value, expand (=spectrum)
    assert len(k0.shape) == 1

    # add N particle dimension
    k0 = k0.unsqueeze(0)

    # input shape r_c,s: N particles
    # input shape eps_env: N wavelengths
    # input shape eps_c,s: (N particles, N wavelengths)
    r_c = torch.as_tensor(r_c)
    r_c = torch.atleast_1d(r_c)  # if single particle, expand
    assert len(r_c.shape) == 1
    r_c = r_c.unsqueeze(-1)  # add wavelength dimension
    r_c = r_c.broadcast_to((r_c.shape[0], k0.shape[1]))

    r_s = torch.as_tensor(r_s)
    r_s = torch.atleast_1d(r_s)  # if single particle, expand
    assert len(r_s.shape) == 1
    r_s = r_s.unsqueeze(-1)  # add wavelength dimension
    r_s = r_s.broadcast_to((r_s.shape[0], k0.shape[1]))
    assert r_c.shape == r_s.shape

    eps_c = torch.as_tensor(eps_c)
    eps_c = torch.atleast_1d(eps_c)
    if eps_c.dim() == 1 and len(eps_c) == len(r_c):
        eps_c = eps_c.broadcast_to((r_c.shape[0], k0.shape[1]))
    else:
        eps_c = eps_c.reshape((r_c.shape[0], k0.shape[1]))

    eps_s = torch.as_tensor(eps_s)
    eps_s = torch.atleast_1d(eps_s)
    if eps_s.dim() == 1 and len(eps_s) == len(r_s):
        eps_s = eps_s.broadcast_to((r_s.shape[0], k0.shape[1]))
    else:
        eps_s = eps_s.reshape((r_s.shape[0], k0.shape[1]))

    assert r_c.shape[0] == r_s.shape[0]
    assert eps_c.shape[0] == r_c.shape[0]
    assert eps_s.shape[0] == r_c.shape[0]
    assert eps_c.shape[1] == k0.shape[1]
    assert eps_s.shape[1] == k0.shape[1]

    # input shape should be as k0
    eps_env = torch.as_tensor(eps_env)
    eps_env = torch.atleast_1d(eps_env).unsqueeze(0)  # particle dim.

    return k0, r_c, r_s, eps_c, eps_s, eps_env


def _get_mie_a_b(
    k0,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="torch",
    precision="double",
    which_jn="recurrence",
    n_max=None,
):
    eps_env = torch.as_tensor(eps_env)

    # core-only: set shell == core
    if r_s is None:
        r_s = r_c

    # core-only: set shell eps == core eps
    if eps_s is None:
        eps_s = eps_c

    k0, r_c, r_s, eps_c, eps_s, eps_env = _broadcast_mie_config(
        k0, r_c, r_s, eps_c, eps_s, eps_env
    )
    n_c = eps_c**0.5
    n_s = eps_s**0.5
    n_env = eps_env**0.5

    # - Mie truncation order
    if n_max is None:
        # automatically determine truncation
        ka = r_s * k0 * torch.sqrt(eps_env)
        n_max = helper.get_truncution_criteroin_wiscombe(ka)
    n_max = torch.as_tensor(n_max, device=k0.device)
    n = torch.arange(1, n_max + 1, device=k0.device)

    # - eval Mie coefficients
    k = k0 * n_env
    x = k * r_c
    y = k * r_s
    m_c = n_c / n_env
    m_s = n_s / n_env
    # this will return order 1 to n_max (no zero order!)
    a_n, b_n = ab(
        x, y, n_max, m_c, m_s, backend=backend, precision=precision, which_jn=which_jn
    )
    return dict(
        k=k,
        k0=k0,
        a_n=a_n,
        b_n=b_n,
        n=n,
        n_max=n_max,
        r_c=r_c,
        r_s=r_s,
        eps_c=eps_c,
        eps_s=eps_s,
        eps_env=eps_env,
        n_c=n_c,
        n_s=n_s,
        n_env=n_env,
    )


# - Observables
def cross_sections(
    k0,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="torch",
    precision="double",
    which_jn="recurrence",
    n_max=None,
) -> dict:
    """compute farfield cross-sections incuding multipole decomposition

    **Caution!** Always returns as second dimension de number of particles (--> 1 if a single particle)

    this function provides autodiff compatible farfield cross-sections calculations,
    they are computed using the analytical solutions provided in:

    Bohren, Craig F., and Donald R. Huffman.
    Absorption and scattering of light by small particles. John Wiley & Sons, 2008.

    Results are retured as a dictionary with keys:
        - 'wavelength' : evaluation wavelengths
        - 'k0' : evaluation wavenumbers
        - 'cs_geo' : geometric cross section
        - 'q_ext' : extiniction efficiency
        - 'q_sca' : scattering efficiency
        - 'q_abs' : absorbtion efficiency
        - 'cs_ext' : extiniction cross section
        - 'cs_sca' : scattering cross section
        - 'cs_abs' : absorbtion cross section
        - 'q_ext_multipoles' : multipole decomp. of extiniction efficiency
        - 'q_sca_multipoles' : multipole decomp. of scattering efficiency
        - 'q_abs_multipoles' : multipole decomp. of absorbtion efficiency
        - 'cs_ext_multipoles' : multipole decomp. of extiniction cross section
        - 'cs_sca_multipoles' : multipole decomp. of scattering cross section
        - 'cs_abs_multipoles' : multipole decomp. of absorbtion cross section

    vectorization needs to follow the conventions (see :func:`_broadcast_mie_config` for details):
        - dimension 0: N particles to calc.
        - dimension 1: spectral dimension (k0)
        - dimension 2: mie-order

    Args:
        k0 (torch.Tensor): evaluation wavenumbers, must be the same for all particles and Mie orders. 1D tensor of shape (N).
        r_c (torch.Tensor): core radius (in nm).
        eps_c (torch.Tensor): permittivity of core.
        r_s (torch.Tensor, optional): shell radius (in nm). Defaults to None.
        eps_s (torch.Tensor, optional): permittivity of shell. Defaults to None.
        eps_env (float, optional): permittivity of environment. Defaults to 1.0.
        backend (str, optional): backend to use for spherical bessel functions. Either 'scipy' or 'torch'. Defaults to 'scipy'.
        precision (str, optional): has no effect on the scipy implementation.
        which_jn (str, optional): only for "torch" backend. Which algorithm for j_n to use. Either 'stable' or 'fast'. Defaults to 'stable'.
        n_max (int, optional): highest order to compute. Defaults to None.

    Returns:
        dict: dict containing all resulting spectra.
    """
    # - evaluate mie coefficients (vectorized)
    miecoeff = _get_mie_a_b(
        k0=k0,
        r_c=r_c,
        eps_c=eps_c,
        r_s=r_s,
        eps_s=eps_s,
        eps_env=eps_env,
        backend=backend,
        precision=precision,
        which_jn=which_jn,
        n_max=n_max,
    )
    n_max = miecoeff["n_max"]
    n = miecoeff["n"].unsqueeze(-1).unsqueeze(-1)  # add dim N_part, N_k0
    k = miecoeff["k"].unsqueeze(0)  # add dim order
    k0 = miecoeff["k0"].unsqueeze(0)  # add dim order
    r_s = miecoeff["r_s"].unsqueeze(0)  # add dim order
    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 * torch.pi / (k**2)

    cs_ext_mp = prefactor * (2 * n + 1) * torch.stack((a_n.real, b_n.real))

    cs_sca_mp = prefactor * (2 * n + 1) * torch.stack((a_n.abs() ** 2, b_n.abs() ** 2))
    cs_abs_mp = cs_ext_mp - cs_sca_mp

    # full cross-sections:
    # sum multipole types (index 0) and multipole orders (index 1)
    cs_ext = torch.sum(cs_ext_mp, (0, 1)).real
    cs_abs = torch.sum(cs_abs_mp, (0, 1)).real
    cs_sca = torch.sum(cs_sca_mp, (0, 1)).real

    return dict(
        wavelength=2 * torch.pi / k0.squeeze(),
        k0=k0.squeeze(),
        cs_geo=cs_geo,
        # full cross sections
        q_ext=cs_ext / cs_geo[0, :],
        q_sca=cs_sca / cs_geo[0, :],
        q_abs=cs_abs / cs_geo[0, :],
        cs_ext=cs_ext,
        cs_sca=cs_sca,
        cs_abs=cs_abs,
        # separate multipoles
        q_ext_multipoles=cs_ext_mp.real / cs_geo,
        q_sca_multipoles=cs_sca_mp.real / cs_geo,
        q_abs_multipoles=cs_abs_mp.real / cs_geo,
        cs_ext_multipoles=cs_ext_mp.real,
        cs_sca_multipoles=cs_sca_mp.real,
        cs_abs_multipoles=cs_abs_mp.real,
    )


def angular_scattering(
    k0,
    theta,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="torch",
    precision="double",
    which_jn="recurrence",
    n_max=None,
) -> dict:
    """compute farfield angular scattering

    this function provides autodiff compatible farfield anglar scattering calculations,
    they are computed using the analytical solutions provided in:

    Bohren, Craig F., and Donald R. Huffman.
    Absorption and scattering of light by small particles. John Wiley & Sons, 2008.

    Results are retured as a dictionary with keys:
        - 'wavelength' : evaluation wavelengths
        - 'k0' : evaluation wavenumbers
        - 'theta' : evaluation angles
        - 'S1' : S1 s parameter
        - 'S2' : S2 s parameter
        - 'i_per' : scattered irradiance per unit incident irradiance for perpendicular light
        - 'i_par' : scattered irradiance per unit incident irradiance for parallel light
        - 'i_unpol' : scattered irradiance per unit incident irradiance for unpolarised light
        - 'pol_degree' : the polarisation factor

    vectorization needs to follow the conventions (see :func:`_broadcast_mie_config` for details):
        - dimension 0: N particles to calc.
        - dimension 1: spectral dimension (k0)
        - dimension 2: mie-order
        - dimension 3: angular resulution

    Args:
        k0 (torch.Tensor): evaluation wavenumbers, must be the same for all particles and Mie orders. 1D tensor of shape (N).
        theta (torch.Tensor): evaluation angles (rad)
        r_c (torch.Tensor): core radius (in nm).
        eps_c (torch.Tensor): permittivity of core.
        r_s (torch.Tensor, optional): shell radius (in nm). Defaults to None.
        eps_s (torch.Tensor, optional): permittivity of shell. Defaults to None.
        eps_env (float, optional): permittivity of environment. Defaults to 1.0.
        backend (str, optional): backend to use for spherical bessel functions. Either 'scipy' or 'torch'. Defaults to 'scipy'.
        precision (str, optional): has no effect on the scipy implementation.
        which_jn (str, optional): only for "torch" backend. Which algorithm for j_n to use. Either 'stable' or 'fast'. Defaults to 'stable'.
        n_max (int, optional): highest order to compute. Defaults to None.

    Returns:
        dict: dict containing all angular scattering results for all wavenumbers and angles
    """
    # - evaluate mie coefficients (vectorized)
    miecoeff = _get_mie_a_b(
        k0, r_c, eps_c, r_s, eps_s, eps_env, backend, precision, which_jn, n_max
    )
    n_max = miecoeff["n_max"]
    n = miecoeff["n"]
    k = miecoeff["k"]
    k0 = miecoeff["k0"]
    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    mu = torch.cos(theta)
    pi_n, tau_n = special.pi_tau(n_max - 1, mu)  # shape: N_teta, n_Mie_order

    # vectorization:
    #   - dim 0: n particles
    #   - dim 1: wavevectors
    #   - dim 2: Mie order
    #   - dim 3: teta angles
    k = k.unsqueeze(0)  # add dim order
    k0 = k0.unsqueeze(0)  # add dim order
    pi_n = pi_n.unsqueeze(1).unsqueeze(1)  # add N_part, k0 dims.
    tau_n = tau_n.unsqueeze(1).unsqueeze(1)  # add N_part, k0 dims.
    n = n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # add dim N_part, N_k0, N_theta
    a_n = a_n.unsqueeze(-1)
    b_n = b_n.unsqueeze(-1)

    # eval. S1 and S2, sum over Mie orders (dim 0)
    s1 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * pi_n + b_n * tau_n), dim=0)
    s2 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * tau_n + b_n * pi_n), dim=0)

    i_per = s1.abs() ** 2
    i_par = s2.abs() ** 2

    i_unpol = (i_par + i_per) / 2
    pol_degree = (i_per - i_par) / (i_per + i_par)

    return dict(
        wavelength=2 * torch.pi / k0.squeeze(),
        k0=k0.squeeze(),
        theta=theta.squeeze(),
        # observables
        S1=s1,
        S2=s2,
        i_per=i_per,
        i_par=i_par,
        i_unpol=i_unpol,
        pol_degree=pol_degree,
    )
