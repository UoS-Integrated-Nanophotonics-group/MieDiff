import warnings
import numpy as np
import torch
from pymiediff import coreshell
from pymiediff import angular
from pymiediff import helper


def cross_sections(
    k0,
    r_c,
    eps_c,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    func_an=coreshell.an,
    func_bn=coreshell.bn,
    n_max=None,
):
    eps_env = torch.as_tensor(eps_env)

    # core-only: set shell == core
    if r_s is None:
        r_s = r_c

    # core-only: set shell eps == core eps
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

    n_c = torch.broadcast_to(torch.atleast_1d(eps_c).unsqueeze(1), k0.shape) ** 0.5
    n_s = torch.broadcast_to(torch.atleast_1d(eps_s).unsqueeze(1), k0.shape) ** 0.5
    n_env = torch.broadcast_to(torch.atleast_1d(eps_env).unsqueeze(1), k0.shape) ** 0.5

    # - Mie truncation order
    if n_max is None:
        # automatically determine truncation
        ka = r_s * k0 * torch.sqrt(eps_env)
        n_max = helper.get_truncution_criteroin_wiscombe(ka)
    n = torch.arange(1, n_max + 1).unsqueeze(0)  # dim. 0: spectral dimension (k0)
    assert len(n.shape) == 2

    # - eval Mie coefficients
    x = k0 * r_c
    y = k0 * r_s
    m_c = n_c / n_env
    m_s = n_s / n_env
    a_n = func_an(x, y, n, m_c, m_s)
    b_n = func_bn(x, y, n, m_c, m_s)

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 * torch.pi / (k0**2)

    cs_ext_mp = prefactor * (2 * n + 1) * torch.stack((a_n.real, b_n.real))

    cs_sca_mp = (
        prefactor
        * (2 * n + 1)
        * torch.stack((a_n.abs() ** 2 + a_n.imag**2, b_n.real**2 + b_n.imag**2))
    )
    cs_abs_mp = cs_ext_mp - cs_sca_mp

    # full cross-sections:
    # sum multipole types (index 0) and multipole orders (index -1)
    cs_ext = torch.sum(cs_ext_mp, (0, -1))
    cs_abs = torch.sum(cs_abs_mp, (0, -1))
    cs_sca = torch.sum(cs_sca_mp, (0, -1))

    return dict(
        cs_geo=cs_geo,
        # full cross sections
        q_ext=cs_ext / cs_geo,
        q_sca=cs_sca / cs_geo,
        q_abs=cs_abs / cs_geo,
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
    an=coreshell.an,
    bn=coreshell.bn,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    n_max=None,
):
    # core-only: set shell == core
    if r_s is None:
        r_s = r_c
    if n_max is None:
        xShell = k0.detach() * r_c
        n_max = int(
            np.round(2 + xShell + 4 * (xShell ** (1 / 3)))
        )  # automatic eval. of adequate n_max.
    n = torch.arange(1, n_max + 1).unsqueeze(0)  # dim. 0: spectral dimension (k0)
    assert len(n.shape) == 2
    # core-only: set shell eps == core eps
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

    n_c = torch.broadcast_to(torch.atleast_1d(eps_c).unsqueeze(1), k0.shape) ** 0.5
    n_s = torch.broadcast_to(torch.atleast_1d(eps_s).unsqueeze(1), k0.shape) ** 0.5
    n_env = torch.broadcast_to(torch.atleast_1d(eps_env).unsqueeze(1), k0.shape) ** 0.5

    # - eval Mie coefficients
    x = k0 * r_c
    y = k0 * r_s
    m_c = n_c / n_env
    m_s = n_s / n_env
    a_n = an(x, y, n, m_c, m_s)
    b_n = bn(x, y, n, m_c, m_s)

    mu = torch.cos(theta)

    pi, tau = angular.pi_tau(n_max - 1, mu)

    s1 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * pi + b_n * tau), dim=1)
    s2 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * tau + b_n * pi), dim=1)

    i_per = s1.abs() ** 2
    i_par = s2.abs() ** 2

    i_unp = (i_par + i_per) / 2
    P = (i_per - i_par) / (i_per + i_par)

    return dict(
        S1=s1,
        S2=s2,
        i_per=i_per,
        i_par=i_par,
        i_unp=i_unp,
        P=P,
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

    multipoles = cross_sections_mp(
        k0=k0,
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
    )

    multipole_sca = multipoles["q_sca"]

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
    i_unp = ang_scattering["i_unp"]
    P = ang_scattering["P"]

    # Plotting
    fig = plt.figure()
    # fig, ax = plt.subplots(2, 2, figsize=(12, 5), dpi=200)

    ax1 = fig.add_subplot(221)
    pmd.helper.plot_cross_section(
        ax1,
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        waveLengths=(2 * np.pi) / k0,
        scattering=(cross_section_sca, cross_section_ext, cross_section_abs),
        names=("sca", "ext", "abs"),
    )

    ax2 = fig.add_subplot(223)
    pmd.helper.plot_cross_section(
        ax2,
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        waveLengths=(2 * np.pi) / k0,
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
        wavelength=(2 * np.pi) / k0_single,
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
        wavelength=(2 * np.pi) / k0_single,
        angles=theta,
        scattering=(s1, s2),
        names=("$S_1$", "$S_2$"),
        title="Corresponding S parameters.",
    )

    fig.tight_layout()
    plt.show()
