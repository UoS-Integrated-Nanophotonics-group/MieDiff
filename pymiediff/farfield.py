import warnings
import numpy as np
import torch
from pymiediff import coreshell
from pymiediff import angular


def cross_sections(k0, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, an = coreshell.an, bn = coreshell.bn, n_max=None):
    # core-only: set shell == core
    if r_s is None:
        r_s = r_c
    if n_max is None:
        # print(k0.shape)
        # xShell = np.max(k0.detach().numpy()) * r_c
        # print(xShell)
        # n_max = int(
        #     np.round(2 + xShell + 4 * (xShell ** (1 / 3)))
        # )  # automatic eval. of adequate n_max.
        n_max = 8
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


def cross_sections_mp(k0, r_c, eps_c, an = coreshell.an, bn = coreshell.bn, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
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
    prefactor = 2 / (k0**2) #* r_s**2)

    q_ext = prefactor * (2 * n + 1) * torch.stack((a_n.real, b_n.real))

    q_sca = (
        prefactor
        * (2 * n + 1)
        * torch.stack((a_n.abs() ** 2 + a_n.imag**2, b_n.real**2 + b_n.imag**2))
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


def angular_scattering(k0, theta, r_c, eps_c, an = coreshell.an, bn = coreshell.bn, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
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