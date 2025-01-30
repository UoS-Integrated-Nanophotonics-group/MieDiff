import torch
import numpy as np
from pymiediff import coreshell


def pi_tau(N, mu):
    # Ensure N is an integer
    N = int(N)

    # Ensure mu is 1D to avoid shape mismatches
    mu = mu.view(-1)

    # Preallocate tensors for π and τ with the correct shape
    pies = torch.zeros(len(mu), N + 1, dtype=mu.dtype, device=mu.device)
    taus = torch.zeros(len(mu), N + 1, dtype=mu.dtype, device=mu.device)

    # Initialize the first two terms
    pies[:, 0] = 1.0  # π_0 = 1
    taus[:, 0] = mu  # τ_0 = μ
    if N > 0:
        pies[:, 1] = 3 * mu  # π_1 = 3 * μ
        taus[:, 1] = 3 * torch.cos(2 * torch.acos(mu))  # τ_1 = 3cos(2cos⁻¹(μ))

    for n in range(2, N + 1):
        # Compute pies[:, n] out of place
        clone_of_pies = pies.clone()
        pi_n = (
            (2 * n + 1) * mu * clone_of_pies[:, n - 1]
            - (n + 1) * clone_of_pies[:, n - 2]
        ) / n
        pies[:, n] = pi_n

        # Compute taus[:, n] out of place
        clone_of_pies = pies.clone()
        tau_n = (n + 1) * mu * clone_of_pies[:, n] - (n + 2) * clone_of_pies[:, n - 1]
        taus[:, n] = tau_n

    return pies, taus


# - dimension 0 will be for angles .unsqueeze(1)
# - dimension 1 will be for mie-order .unsqueeze(0)


def smat(k0, theta, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
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
    a_n = coreshell.an(x, y, n, m_c, m_s)
    b_n = coreshell.bn(x, y, n, m_c, m_s)

    mu = torch.cos(theta)

    pi, tau = pi_tau(n_max - 1, mu)

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
