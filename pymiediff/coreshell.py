import warnings

import numpy as np
import torch

# from . import special
from pymiediff import special  # use absolute package internal imports!
from pymiediff.special import psi, psi_der, chi, chi_der, xi, xi_der


def expressions():
    return ["q_ext", "q_sca", "q_abs", "cs_ext", "cs_sca", "cs_abs"]


def An(x, n, m1, m2):
    return (
        m2 * psi(m2 * x, n) * psi_der(m1 * x, n)
        - m1 * psi_der(m2 * x, n) * psi(m1 * x, n)
    ) / (
        m2 * chi(m2 * x, n) * psi_der(m1 * x, n)
        - m1 * chi_der(m2 * x, n) * psi(m1 * x, n)
    )


def Bn(x, n, m1, m2):
    return (
        m2 * psi(m1 * x, n) * psi_der(m2 * x, n)
        - m1 * psi(m2 * x, n) * psi_der(m1 * x, n)
    ) / (
        m2 * chi_der(m2 * x, n) * psi(m1 * x, n)
        - m1 * psi_der(m1 * x, n) * chi(m2 * x, n)
    )


def an(x, y, n, m1, m2):
    return (
        psi(y, n) * (psi_der(m2 * y, n) - An(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * psi_der(y, n) * (psi(m2 * y, n) - An(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        xi(y, n) * (psi_der(m2 * y, n) - An(x, n, m1, m2) * chi_der(m2 * y, n))
        - m2 * xi_der(y, n) * (psi(m2 * y, n) - An(x, n, m1, m2) * chi(m2 * y, n))
    )


def bn(x, y, n, m1, m2):
    return (
        m2 * psi(y, n) * (psi_der(m2 * y, n) - Bn(x, n, m1, m2) * chi_der(m2 * y, n))
        - psi_der(y, n) * (psi(m2 * y, n) - Bn(x, n, m1, m2) * chi(m2 * y, n))
    ) / (
        m2 * xi(y, n) * (psi_der(m2 * y, n) - Bn(x, n, m1, m2) * chi_der(m2 * y, n))
        - xi_der(y, n) * (psi(m2 * y, n) - Bn(x, n, m1, m2) * chi(m2 * y, n))
    )


def scs(k0, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
    # core-only: set shell == core
    if r_s is None:
        r_s = r_c
    if n_max is None:
        xShell = torch.max(k0.detach()) * r_c
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

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 * torch.pi / (k0**2)  # * r_s**2)

    q_ext = torch.sum(prefactor * (2 * n + 1) * ((a_n + b_n).real), dim=1)

    q_sca = torch.sum(
        prefactor
        * (2 * n + 1)
        * (a_n.real**2 + a_n.imag**2 + b_n.real**2 + b_n.imag**2),
        dim=1,
    )

    q_abs = q_ext - q_sca

    return dict(
        q_ext=q_ext / cs_geo,
        q_sca=q_sca / cs_geo,
        q_abs=q_abs / cs_geo,
        cs_geo=cs_geo,
        cs_ext=q_ext,
        cs_sca=q_sca,
        cs_abs=q_abs,
    )


def scs_mp(k0, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
    # core-only: set shell == core
    if r_s is None:
        r_s = r_c
    if n_max is None:
        xShell = torch.max(k0.detach()) * r_c
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

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 / (k0**2 * r_s**2)

    q_ext = prefactor * (2 * n + 1) * torch.stack((a_n.real, b_n.real))

    q_sca = (
        prefactor
        * (2 * n + 1)
        * torch.stack((a_n.abs() ** 2 + a_n.imag**2, b_n.real**2 + b_n.imag**2))
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
    starting_wavelength = 200  # nm
    ending_wavelength = 600  # nm
    # ===================================

    dtype = torch.complex128  # torch.complex64
    device = torch.device("cpu")

    k0 = (
        2 * torch.pi / torch.linspace(starting_wavelength, ending_wavelength, N_pt_test)
    )

    r_c = torch.tensor(core_radius)
    r_s = torch.tensor(shell_radius)
    n_c = torch.tensor(core_refractiveIndex)
    n_s = torch.tensor(shell_refractiveIndex)

    t0 = time.time()

    cross_section = pmd.coreshell.scs(
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

    multipoles = pmd.coreshell.scs_mp(
        k0=k0,
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
    )

    multipole_sca = multipoles["q_sca"]

    # Plotting

    fig, ax = plt.subplots(2, figsize=(12, 5), dpi=200)

    pmd.helper.PlotCrossSection(
        ax[0],
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        waveLengths=(2 * np.pi) / k0,
        scattering=(cross_section_sca, cross_section_ext, cross_section_abs),
        names=("sca", "ext", "abs"),
    )

    pmd.helper.PlotCrossSection(
        ax[1],
        radi=(r_c, r_s),
        ns=(n_c, n_s),
        waveLengths=(2 * np.pi) / k0,
        scattering=cross_section_sca,
        names="sca",
        multipoles=multipole_sca,
        max_dis=4,
        title="Multipole decomp.",
    )

    fig.tight_layout()

    import PyMieScatt as pms

    wl = np.linspace(starting_wavelength, ending_wavelength, N_pt_test)

    cexts = []
    cscas = []
    cabss = []

    for w in wl:
        cext, csca, cabs, _, _, _, _ = pms.MieQCoreShell(
            core_refractiveIndex,
            shell_refractiveIndex,
            w,
            core_radius * 2,
            shell_radius * 2,
        )

        cexts.append(cext)
        cscas.append(csca)
        cabss.append(cabs)

    ax[0].plot(wl, cexts, label="pms cext", linestyle="--", linewidth=1)
    ax[0].plot(wl, cscas, label="pms csca", linestyle="--", linewidth=1)
    ax[0].plot(wl, cabss, label="pms cabs", linestyle="--", linewidth=1)
    ax[0].legend()

    plt.show()
