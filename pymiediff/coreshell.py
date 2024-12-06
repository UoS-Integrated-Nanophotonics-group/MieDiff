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

    q_ext = torch.sum(prefactor * (2 * n + 1) * ((a_n + b_n).real), dim=1)

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


def scs_mp(k0, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
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

    N_pt_test = 1000
    N_order_test = 6

    dtype = torch.complex128  # torch.complex64
    device = torch.device("cpu")

    k0 = 2 * torch.pi / torch.linspace(200, 600, N_pt_test)

    r_c = torch.tensor(12.0)
    r_s = torch.tensor(50.0)
    n_c = torch.tensor(2.0 + 0.2j)
    n_s = torch.tensor(5.0 + 0.1j)

    t0 = time.time()

    cross_section = pmd.coreshell.scs(
        k0=k0,
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
        n_max=N_order_test,
    )

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
        n_max=N_order_test,
    )

    multipole_sca = multipoles["q_sca"]

    t1 = time.time()

    print("normal:", cross_section_sca.shape)
    print("mutipole:", multipole_sca.shape)

    time_torch = t1 - t0

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
        max_dis=3,
        title="Multipole decomp.",
    )
    print("Time taken:", time_torch)
    fig.tight_layout()
    plt.show()
