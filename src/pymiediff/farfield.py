# -*- coding: utf-8 -*-
"""
farfield observables

"""
import warnings
import torch
from pymiediff import coreshell
from pymiediff import special
from pymiediff import helper


def _broadcast_mie_config(k0, r_c, r_s, eps_c, eps_s, eps_env):
    """broadcast configs to dimension for vectorization

    dimension convention: (N particles, N wavevectors, N Mie order)

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
    k0 = torch.atleast_1d(k0)  # if single value, expand
    assert len(k0.shape) == 1

    # input shape k: N wavelengths
    k0 = k0.unsqueeze(-1).unsqueeze(0)

    # input shape r_c,s: N particles
    # input shape eps_env: N wavelengths
    # input shape eps_c,s: (N particles, N wavelengths)
    r_c = torch.as_tensor(r_c)
    r_c = torch.atleast_1d(r_c)  # if single particle, expand
    assert len(r_c.shape) == 1
    r_c = r_c.unsqueeze(-1).unsqueeze(-1)
    r_s = torch.as_tensor(r_s)
    r_s = torch.atleast_1d(r_s)  # if single particle, expand
    assert len(r_s.shape) == 1
    r_s = r_s.unsqueeze(-1).unsqueeze(-1)
    assert r_c.shape == r_s.shape

    # input shape should be: [N particles, N k0]
    eps_c = torch.as_tensor(eps_c)
    eps_c = torch.atleast_1d(eps_c)
    if eps_c.dim() == 1 and len(eps_c) == len(r_c):
        eps_c = eps_c.broadcast_to((r_c.shape[0], k0.shape[1], 1))
    else:
        eps_c = eps_c.reshape((r_c.shape[0], k0.shape[1], 1))

    eps_s = torch.as_tensor(eps_s)
    eps_s = torch.atleast_1d(eps_s)
    if eps_s.dim() == 1 and len(eps_s) == len(r_s):
        eps_s = eps_s.broadcast_to((r_s.shape[0], k0.shape[1], 1))
    else:
        eps_s = eps_s.reshape((r_s.shape[0], k0.shape[1], 1))

    assert eps_c.shape[0] == r_c.shape[0]
    assert eps_s.shape[0] == r_c.shape[0]
    assert eps_c.shape[1] == k0.shape[1]
    assert eps_s.shape[1] == k0.shape[1]

    # input shape should be as k0
    eps_env = torch.as_tensor(eps_env)
    eps_env = torch.atleast_1d(eps_env).unsqueeze(0).unsqueeze(-1)

    return k0, r_c, r_s, eps_c, eps_s, eps_env


def cross_sections(
    k0,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="scipy",
    n_max=None,
) -> dict:
    """compute farfield cross-sections incuding multipole decomposition

    **Caution!** Always returns as first dimension de number of particles (--> 1 if a single particle)

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
    n = torch.arange(1, n_max + 1).broadcast_to(
        *k0.shape[:1], -1
    )  # dim 0: N particles, dim. 1: spectral dimension (k0)
    n = n.to(k0.device)

    # - eval Mie coefficients
    x = k0 * r_c
    y = k0 * r_s
    m_c = n_c / n_env
    m_s = n_s / n_env
    a_n, b_n = coreshell.ab(x, y, n, m_c, m_s, backend=backend)

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 * torch.pi / (k0**2)

    cs_ext_mp = prefactor * (2 * n + 1) * torch.stack((a_n.real, b_n.real))

    cs_sca_mp = prefactor * (2 * n + 1) * torch.stack((a_n.abs() ** 2, b_n.abs() ** 2))
    cs_abs_mp = cs_ext_mp - cs_sca_mp

    # full cross-sections:
    # sum multipole types (index 0) and multipole orders (index -1)
    cs_ext = torch.sum(cs_ext_mp, (0, -1))
    cs_abs = torch.sum(cs_abs_mp, (0, -1))
    cs_sca = torch.sum(cs_sca_mp, (0, -1))

    return dict(
        wavelength=2 * torch.pi / k0.squeeze(),
        k0=k0.squeeze(),
        cs_geo=cs_geo,
        # full cross sections
        q_ext=cs_ext / cs_geo[:, 0],
        q_sca=cs_sca / cs_geo[:, 0],
        q_abs=cs_abs / cs_geo[:, 0],
        cs_ext=cs_ext,
        cs_sca=cs_sca,
        cs_abs=cs_abs,
        # separate multipoles
        q_ext_multipoles=cs_ext_mp / cs_geo,
        q_sca_multipoles=cs_sca_mp / cs_geo,
        q_abs_multipoles=cs_abs_mp / cs_geo,
        cs_ext_multipoles=cs_ext_mp,
        cs_sca_multipoles=cs_sca_mp,
        cs_abs_multipoles=cs_abs_mp,
    )


def angular_scattering(
    k0,
    theta,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="scipy",
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
        n_max (int, optional): highest order to compute. Defaults to None.

    Returns:
        dict: dict containing all angular scattering results for all wavenumbers and angles
    """
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
    n = torch.arange(1, n_max + 1).broadcast_to(
        *k0.shape[:1], -1
    )  # dim 0: N particles, dim. 1: spectral dimension (k0)
    n = n.to(k0.device)

    # - eval Mie coefficients
    x = k0 * r_c
    y = k0 * r_s
    m_c = n_c / n_env
    m_s = n_s / n_env
    a_n, b_n = coreshell.ab(x, y, n, m_c, m_s, backend=backend)

    mu = torch.cos(theta)
    pi, tau = special.pi_tau(n_max - 1, mu)  # shape: N_teta, n_Mie_order

    # vectorization:
    #   - dim 0: n particles
    #   - dim 1: wavevectors
    #   - dim 2: Mie order
    #   - dim 3: teta angles
    pi = pi.movedim(0, 1).unsqueeze(0).unsqueeze(0)
    tau = tau.movedim(0, 1).unsqueeze(0).unsqueeze(0)
    n = n.unsqueeze(-1)
    a_n = a_n.unsqueeze(-1)
    b_n = b_n.unsqueeze(-1)

    # eval. S1 and S2, sum over Mie orders (dim 2)
    s1 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * pi + b_n * tau), dim=2)
    s2 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * tau + b_n * pi), dim=2)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import pymiediff as pmd

    # ======== Test Parameters ==========
    N_pt_test = 1000
    # N_order_test = 6

    # Test Case for testing:
    core_radius = 12.0  # nm
    shell_radius = 50.0  # nm
    core_refractiveIndex = 2.0 + 0.1j
    shell_refractiveIndex = 5.0 + 0.2j
    starting_wavelength = 200.0  # nm
    ending_wavelength = 600.0  # nm
    # Wavelegth for anular
    target_wavelength = 500.0  # nm
    # ===================================

    dtype = torch.complex128  # torch.complex64
    device = torch.device("cpu")

    k0 = (
        2 * torch.pi / torch.linspace(starting_wavelength, ending_wavelength, N_pt_test)
    )

    k0_single = torch.tensor(2 * torch.pi / target_wavelength)

    theta = torch.linspace(0.001, 2 * torch.pi - 0.001, N_pt_test, dtype=torch.double)
    r_c = torch.tensor(core_radius)
    r_s = torch.tensor(shell_radius)
    n_c = torch.tensor(core_refractiveIndex)
    n_s = torch.tensor(shell_refractiveIndex)

    t0 = time.time()

    cross_section = cross_sections(
        k0=k0,
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
    )

    t1 = time.time()
    time_torch = t1 - t0
    print("Time taken:", time_torch)

    cross_section_sca = cross_section["q_sca"]
    cross_section_ext = cross_section["q_ext"]
    cross_section_abs = cross_section["q_abs"]
    cross_section_geo = cross_section["cs_geo"]

    multipole_sca = cross_section["q_sca_multipoles"]

    ang_scattering = angular_scattering(
        k0=k0_single,
        theta=theta,
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
    )

    s1 = ang_scattering["S1"]
    s2 = ang_scattering["S2"]
    i_per = ang_scattering["i_per"]
    i_par = ang_scattering["i_par"]
    i_unp = ang_scattering["i_unpol"]
    P = ang_scattering["pol_degree"]

    # Plotting
    fig = plt.figure()
    # fig, ax = plt.subplots(2, 2, figsize=(12, 5), dpi=200)

    ax1 = fig.add_subplot(221)
    pmd.helper.plot_cross_section(
        ax1,
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        waveLengths=(2 * torch.pi) / k0,
        scattering=(cross_section_sca, cross_section_ext, cross_section_abs),
        names=("sca", "ext", "abs"),
    )

    ax2 = fig.add_subplot(223)
    pmd.helper.plot_cross_section(
        ax2,
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        waveLengths=(2 * torch.pi) / k0,
        scattering=cross_section_sca,
        names="sca",
        multipoles=multipole_sca,
        max_dis=4,
        title="Multipole decomp.",
    )

    ax3 = fig.add_subplot(222, projection="polar")
    pmd.helper.plot_angular(
        ax3,
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        wavelength=(2 * torch.pi) / k0_single,
        angles=theta,
        scattering=(i_per, i_par, i_unp),
        names=("$i_{per}$", "$i_{par}$", "$i_{unp}$"),
        title=f"Scattered irradiance per unit incident irradiance at $\lambda = {target_wavelength}$.",
    )

    ax4 = fig.add_subplot(224, projection="polar")
    pmd.helper.plot_angular(
        ax4,
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        wavelength=(2 * torch.pi) / k0_single,
        angles=theta,
        scattering=(s1, s2),
        names=("$S_1$", "$S_2$"),
        title="Corresponding S parameters.",
    )

    fig.tight_layout()
    plt.show()
