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


# - Mie coefficients, private API
def _miecoef(
    x,
    y,
    n,
    m1,
    m2,
    return_internal=False,
    backend="torch",
    precision="double",
    which_jn="recurrence",
):
    """an and bn scattering coefficients

    native torch implementation via bessel upward and downward recurrences

    Bohren & Huffman: Absorption and scattering of light by small particles.
    Pg. 183 and Eq. 8.1. Solved using sympy.

    Args:
        x (torch.Tensor): size parameter (core)
        y (torch.Tensor): size parameter (shell)
        n (torch.Tensor): orders to compute
        m1 (torch.Tensor): relative refractive index of core
        m2 (torch.Tensor): relative refractive index of shell
        return_internal (float, optional): If True, return also internal Mie coefficients (longer computation time). Defaults to False.
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

    # permeabilities are set to 1 for now
    mu1 = mu2 = mu = 1.0

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

    # # Mie coeffs.
    # An = (m2 * psi_m2x * dpsi_m1x - m1 * dpsi_m2x * psi_m1x) / (
    #     m2 * chi_m2x * dpsi_m1x - m1 * dchi_m2x * psi_m1x
    # )
    # Bn = (m2 * psi_m1x * dpsi_m2x - m1 * psi_m2x * dpsi_m1x) / (
    #     m2 * dchi_m2x * psi_m1x - m1 * dpsi_m1x * chi_m2x
    # )

    # an = (
    #     psi_y * (dpsi_m2y - An * dchi_m2y) - m2 * dpsi_y * (psi_m2y - An * chi_m2y)
    # ) / (xi_y * (dpsi_m2y - An * dchi_m2y) - m2 * dxi_y * (psi_m2y - An * chi_m2y))
    # bn = (
    #     m2 * psi_y * (dpsi_m2y - Bn * dchi_m2y) - dpsi_y * (psi_m2y - Bn * chi_m2y)
    # ) / (m2 * xi_y * (dpsi_m2y - Bn * dchi_m2y) - dxi_y * (psi_m2y - Bn * chi_m2y))

    # common expressions - scattering coefficients
    # (Via sympy, solving Bohren Huffmann Eq 8.1)
    x0 = dpsi_m2y * psi_y
    x1 = chi_m2x * dpsi_m1x
    x2 = m2 * mu2
    x3 = mu1 * x2
    x4 = x1 * x3
    x5 = m1 * mu2**2
    x6 = dpsi_m2x * psi_m1x
    x7 = dchi_m2y * psi_y
    x8 = x6 * x7
    x9 = dpsi_m1x * psi_m2x
    x10 = x3 * x9
    x11 = dchi_m2x * psi_m1x
    x12 = psi_m2y * x11
    x13 = m1 * x2
    x14 = mu * x13
    x15 = x12 * x14
    x16 = x11 * x5
    x17 = m2**2 * mu1
    x18 = x17 * x9
    x19 = chi_m2y * dpsi_y
    x20 = mu * x19
    x21 = x13 * x6
    x22 = chi_m2x * psi_m2y
    x23 = mu * x17
    x24 = dpsi_m1x * x22 * x23
    x25 = dpsi_m2y * xi_y
    x26 = dchi_m2y * xi_y
    x27 = x26 * x6
    x28 = chi_m2y * dxi_y
    x29 = mu * x28
    x30 = 1 / (
        dxi_y * x15
        - dxi_y * x24
        - x10 * x26
        - x16 * x25
        + x18 * x29
        - x21 * x29
        + x25 * x4
        + x27 * x5
    )
    x31 = x12 * x3
    x32 = dpsi_m1x * x5
    x33 = psi_m2x * x32
    x34 = x3 * x6
    x35 = x1 * x14
    x36 = x22 * x32
    x37 = x14 * x9
    x38 = x11 * x23
    x39 = 1 / (
        dxi_y * x31
        - dxi_y * x36
        + x23 * x27
        + x25 * x35
        - x25 * x38
        - x26 * x37
        + x28 * x33
        - x28 * x34
    )

    # scattering coefficients (external)
    an = x30 * (
        dpsi_y * x15
        - dpsi_y * x24
        - x0 * x16
        + x0 * x4
        - x10 * x7
        + x18 * x20
        - x20 * x21
        + x5 * x8
    )
    bn = x39 * (
        dpsi_y * x31
        - dpsi_y * x36
        + x0 * x35
        - x0 * x38
        + x19 * x33
        - x19 * x34
        + x23 * x8
        - x37 * x7
    )

    if return_internal:
        # common expressions - internal coefficients
        x40 = x2 * x39
        x41 = chi_m2x * dpsi_m2x
        x42 = dpsi_y * xi_y
        x43 = dxi_y * psi_y
        x44 = dchi_m2x * psi_m2x
        x45 = m1 * mu1 * (x41 * x42 - x41 * x43 - x42 * x44 + x43 * x44)
        x46 = x2 * x30

        x47 = m1 * mu2
        x48 = x1 * x42
        x49 = m2 * mu1
        x50 = x11 * x43
        x51 = x1 * x43
        x52 = x11 * x42
        x53 = x42 * x47
        x54 = x49 * x6
        x55 = x43 * x47
        x56 = x49 * x9

        # internal coefficients (core)
        cn = x40 * x45
        dn = x45 * x46

        # internal coefficients (shell)
        fn = x40 * (x47 * x48 - x47 * x51 + x49 * x50 - x49 * x52)
        gn = x46 * (x47 * x50 - x47 * x52 + x48 * x49 - x49 * x51)
        vn = x40 * (-x42 * x54 + x43 * x54 + x53 * x9 - x55 * x9)
        wn = x46 * (x42 * x56 - x43 * x56 - x53 * x6 + x55 * x6)

    # recurrences return n+1 orders: remove zeroth order
    if return_internal:
        result_dict = dict(
            a_n=an[1:, ...],
            b_n=bn[1:, ...],
            c_n=cn[1:, ...],
            d_n=dn[1:, ...],
            f_n=fn[1:, ...],
            g_n=gn[1:, ...],
            v_n=vn[1:, ...],
            w_n=wn[1:, ...],
        )
    else:
        result_dict = dict(
            a_n=an[1:, ...],
            b_n=bn[1:, ...],
        )

    return result_dict


# - internal helper
def _broadcast_mie_config(k0, r_c, r_s, eps_c, eps_s, eps_env):
    """broadcast configs to 2 dimensions for vectorization

    dimension convention is (n Mie order, N particles, N wavevectors).
    This function broadcasts all parameters to dimension (N particles, N wavevectors).

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


# - Mie coefficients - public API
def mie_coefficients(
    k0,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    return_internal=False,
    backend="torch",
    precision="double",
    which_jn="recurrence",
    n_max=None,
):
    """compute mie coefficients for a core-shell sphere

    This function returns Mie coefficient broadcasted to
    shape (n Mie order, N particles, N wavevectors).

    Bohren, Craig F., and Donald R. Huffman.
    Absorption and scattering of light by small particles. John Wiley & Sons, 2008.
    Eqs. 8.1

    Results are retured as a dictionary with keys:
        - 'a_n' : external electric Mie coefficient
        - 'b_n' : external magnetic Mie coefficient
        - 'k0' : evaluation wavenumbers
        - 'k' : evaluation wavenumbers in host medium
        - 'n' : mie orders
        - 'n_max' : maximum mie order
        - 'r_c' : core radius
        - 'r_s' : shell radius
        - 'eps_c' : core permittivities
        - 'eps_s' : shell permittivities
        - 'eps_env' : environmental permittivity
        - 'n_c' : core refractive index
        - 'n_s' : shell refractive index
        - 'n_env' : environmental refractive index
    if kwarg `return_internal` is True, the returned dict contains also:
        - 'c_n' : internal magnetic Mie coefficient (core)
        - 'd_n' : internal electric Mie coefficient (core)
        - 'f_n' : internal magnetic Mie coefficient - first kind (shell)
        - 'g_n' : internal electric Mie coefficient - first kind (shell)
        - 'v_n' : internal magnetic Mie coefficient - second kind (shell)
        - 'w_n' : internal electric Mie coefficient - second kind (shell)


    Args:
        k0 (torch.Tensor): evaluation wavenumbers, must be the same for all particles and Mie orders. 1D tensor of shape (N).
        r_c (torch.Tensor): core radius (in nm).
        eps_c (torch.Tensor): permittivity of core.
        r_s (torch.Tensor, optional): shell radius (in nm). Defaults to None.
        eps_s (torch.Tensor, optional): permittivity of shell. Defaults to None.
        eps_env (float, optional): permittivity of environment. Defaults to 1.0.
        return_internal (float, optional): If True, return also internal Mie coefficients (longer computation time). Defaults to False.
        backend (str, optional): backend to use for spherical bessel functions. Either 'scipy' or 'torch'. Defaults to 'scipy'.
        precision (str, optional): has no effect on the scipy implementation.
        which_jn (str, optional): only for "torch" backend. Which algorithm for j_n to use. Either 'stable' or 'fast'. Defaults to 'stable'.
        n_max (int, optional): highest order to compute. Defaults to None.

    Returns:
        dict: dict containing all resulting spectra.
    """
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
    mie_coef_result = _miecoef(
        x=x,
        y=y,
        n=n_max,
        m1=m_c,
        m2=m_s,
        return_internal=return_internal,
        backend=backend,
        precision=precision,
        which_jn=which_jn,
    )

    return_dict = dict(
        k=k,
        k0=k0,
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

    for k in mie_coef_result:
        return_dict[k] = mie_coef_result[k]

    return return_dict


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
        - dimension 0: mie-order
        - dimension 1: N particles to calc.
        - dimension 2: spectral dimension (k0)

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
    miecoeff = mie_coefficients(
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
        - dimension 0: mie-order
        - dimension 1: N particles to calc.
        - dimension 2: spectral dimension (k0)
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
    miecoeff = mie_coefficients(
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
    n = miecoeff["n"]
    k = miecoeff["k"]
    k0 = miecoeff["k0"]
    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    mu = torch.cos(theta)
    pi_n, tau_n = special.pi_tau(n_max, mu)  # shape: N_teta, n_Mie_order
    pi_n = pi_n[1:]  # Mie orders 1 - n (no zero order)
    tau_n = tau_n[1:]  # Mie orders 1 - n (no zero order)

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


def nearfields(
    k0,
    r_probe,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    E_0=1,
    backend="torch",
    precision="double",
    which_jn="recurrence",
    n_max=None,
):
    """near fields in and around core-shell particles


    Results are retured as a dictionary with keys:
        - 'E_i': incident E-field
        - 'H_i': incident H-field
        - 'E_s': scattered E-field
        - 'H_s': scattered H-field
        - 'E_t': total E-field
        - 'H_t': total H-field

    Args:
        k0 (_type_): _description_
        r_probe (_type_): _description_
        r_c (_type_): _description_
        eps_c (_type_): _description_
        r_s (_type_, optional): _description_. Defaults to None.
        eps_s (_type_, optional): _description_. Defaults to None.
        eps_env (float, optional): _description_. Defaults to 1.0.
        E_0 (int, optional): _description_. Defaults to 1.
        backend (str, optional): _description_. Defaults to "torch".
        precision (str, optional): _description_. Defaults to "double".
        which_jn (str, optional): _description_. Defaults to "recurrence".
        n_max (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: contains incident, scattered and total E- and H-fields
    """
    from pymiediff.special import vsh
    from pymiediff.helper import transform_xyz_to_spherical
    from pymiediff.helper import transform_fields_spherical_to_cartesian

    # - evaluate mie coefficients
    miecoeff = mie_coefficients(
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
        return_internal=True,
    )

    n = miecoeff["n"]
    n_max = miecoeff["n_max"]
    k = miecoeff["k"]
    k0 = miecoeff["k0"]
    r_c = miecoeff["r_c"]
    r_s = miecoeff["r_s"]

    n_env = miecoeff["n_env"]
    n_sourrounding = n_env
    n_c = miecoeff["n_c"]
    n_core = n_c
    n_s = miecoeff["n_s"]
    n_shell = n_s

    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]
    c_n = miecoeff["c_n"]
    d_n = miecoeff["d_n"]
    f_n = miecoeff["f_n"]
    g_n = miecoeff["g_n"]
    v_n = miecoeff["v_n"]
    w_n = miecoeff["w_n"]

    kc = miecoeff["k0"] * r_c
    ks = miecoeff["k0"] * r_s

    # - convert Cartesian to spherical coordinates
    r, theta, phi = transform_xyz_to_spherical(
        r_probe[..., 0], r_probe[..., 1], r_probe[..., 2]
    )

    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)
    assert n_max >= 0

    # vectorization:
    #   - dim 0: Mie order
    #   - dim 1: n particles
    #   - dim 2: wavevectors
    #   - dim 3: positions
    #   - dim 4: field vector components (3)
    n_p = r_c.shape[0]
    n_k0 = k0.shape[1]
    n_pos = theta.shape[0]
    full_shape = (n_max, n_p, n_k0, n_pos, 3)

    # expand dimensions
    # add order, position, vector dim
    k = k.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    k0 = k0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    kc = kc.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    ks = ks.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    n_sourrounding = n_sourrounding.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # add position, vector dim
    a_n = a_n.unsqueeze(-1).unsqueeze(-1)
    b_n = b_n.unsqueeze(-1).unsqueeze(-1)
    c_n = c_n.unsqueeze(-1).unsqueeze(-1)
    d_n = d_n.unsqueeze(-1).unsqueeze(-1)
    f_n = f_n.unsqueeze(-1).unsqueeze(-1)
    g_n = g_n.unsqueeze(-1).unsqueeze(-1)
    v_n = v_n.unsqueeze(-1).unsqueeze(-1)
    w_n = w_n.unsqueeze(-1).unsqueeze(-1)

    # add order, particle, wavenumber, vector dimensions
    r = r.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    idx_1 = r <= r_c  # positions in core
    idx_2 = torch.logical_and(r_c < r, r <= r_s)  # positions in core
    idx_3 = r > r_s  # outside positions

    phi = phi.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    theta = theta.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    # add particle, wavenumber, position, vector dimensions
    n = n.view((-1,) + (r.ndim - 1) * (1,))

    # evaluate vector spherical harmonics
    # Note: this is not optimum as all VSH are evaluated for all positions
    # TODO: check positions first (inside core, inside shell, outside)
    M1_o1n_c, M1_e1n_c, N1_o1n_c, N1_e1n_c = vsh(
        n_max, k0, n_core, r, theta, phi, kind=1
    )
    M1_o1n_s, M1_e1n_s, N1_o1n_s, N1_e1n_s = vsh(
        n_max, k0, n_shell, r, theta, phi, kind=1
    )
    M2_o1n_s, M2_e1n_s, N2_o1n_s, N2_e1n_s = vsh(
        n_max, k0, n_shell, r, theta, phi, kind=2
    )
    M3_o1n, M3_e1n, N3_o1n, N3_e1n = vsh(
        n_max, k0, n_sourrounding, r, theta, phi, kind=3
    )

    # - scattered fields (Bohren Huffmann, Eq. 4.40, 4.45, 8.0)
    # with En = i^n E0 (2n+1)/(n(n+1)):
    # Es = sum_n En (i a_n N3e1n - b_n M3o1n)
    # Hs = k/(omega mu) sum_n En (i b_n N3o1n + a_n M3e1n)
    # the resulting fields are the spherical coordinate components

    En = 1j**n * E_0 * (2 * n + 1) / (n * (n + 1))
    idx_1 = torch.broadcast_to(idx_1, full_shape)
    idx_2 = torch.broadcast_to(idx_2, full_shape)
    idx_3 = torch.broadcast_to(idx_3, full_shape)

    # electric fields (relative to E0)
    Es_1 = En * (c_n * M1_o1n_c - 1j * d_n * N1_e1n_c)
    Es_2 = En * (
        (f_n * M1_o1n_s - 1j * g_n * N1_e1n_s) - (v_n * M2_o1n_s - 1j * w_n * N2_e1n_s)
    )
    Es_3 = En * (1j * a_n * N3_e1n - b_n * M3_o1n)

    Es = torch.zeros(full_shape, dtype=a_n.dtype, device=a_n.device)
    Es[idx_1] = Es_1[idx_1]
    Es[idx_2] = Es_2[idx_2]
    Es[idx_3] = Es_3[idx_3]

    # magnetic fields (relative to H0)
    Hs_1 = -n_core * En * (d_n * M1_e1n_c + 1j * c_n * N1_o1n_c)
    Hs_2 = (
        -n_shell
        
        * En
        * (
            (g_n * M1_e1n_s + 1j * f_n * N1_o1n_s)
            - (w_n * M2_e1n_s + 1j * v_n * N2_o1n_s)
        )
    )
    Hs_3 = En*n_env * (1j * b_n * N3_o1n + a_n * M3_e1n)

    Hs = torch.zeros(full_shape, dtype=a_n.dtype, device=a_n.device)
    Hs[idx_1] = Hs_1[idx_1]
    Hs[idx_2] = Hs_2[idx_2]
    Hs[idx_3] = Hs_3[idx_3]

    # convert to Cartesian
    Es_xyz = transform_fields_spherical_to_cartesian(
        Es[..., 0], Es[..., 1], Es[..., 2], r[..., 0], theta[..., 0], phi[..., 0]
    )
    Es_xyz = torch.stack(Es_xyz, dim=-1)

    Hs_xyz = transform_fields_spherical_to_cartesian(
        Hs[..., 0], Hs[..., 1], Hs[..., 2], r[..., 0], theta[..., 0], phi[..., 0]
    )
    Hs_xyz = torch.stack(Hs_xyz, dim=-1)

    # sum Mie orders
    Es_xyz = Es_xyz.sum(dim=0)
    Hs_xyz = Hs_xyz.sum(dim=0)

    # incident field: X-pol. plane wave
    # expansion (B&H Eq. 4.37)
    # E_pw = E0 * sum_i [ i^n * (2n+1) / (n(n+1)) * ( M1_o1n - i * N1_e1n ) ]
    # H_pw = (E0 / eta) * sum_i [ i^n * (2n+1) / (n(n+1)) * ( M1_e1n + i * N1_o1n ) ]
    # E0 = En * (M1_o1n_c - 1j * N1_e1n_c)
    # H0 = En * (M1_e1n_c + 1j * N1_o1n_c)
    Ei = torch.zeros_like(Es_xyz)
    Ei[..., 0] = (E_0 * torch.exp(1j * k * r * torch.cos(theta)))[..., 0]

    Hi = torch.zeros_like(Es_xyz)
    Hi[..., 1] = (E_0*n_env * torch.exp(1j * k * r * torch.cos(theta)))[..., 0]

    # add incident field to outside positions
    Etot = Es_xyz.clone()
    Htot = Hs_xyz.clone()
    Etot[idx_3[0, ...]] += Ei[idx_3[0, ...]]
    Htot[idx_3[0, ...]] += Hi[idx_3[0, ...]]

    return_dict = dict(
        E_i=Ei,
        H_i=Hi,
        E_s=Es_xyz,
        H_s=Hs_xyz,
        E_t=Etot,
        H_t=Htot,
    )
    return return_dict
