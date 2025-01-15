import warnings

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
    taus[:, 0] = mu   # τ_0 = μ
    if N > 0:
        pies[:, 1] = 3 * mu  # π_1 = 3 * μ
        # print(3 * torch.cos(2 * torch.acos(mu)))
        taus[:, 1] = 3 * torch.cos(2 * torch.acos(mu))  # τ_1 = 3cos(2cos⁻¹(μ))

    for n in range(2, N + 1):
        # Compute pies[:, n] out of place
        clone_of_pies = pies.clone()
        pi_n = ((2 * n + 1) * mu * clone_of_pies[:, n - 1] - (n + 1) * clone_of_pies[:, n - 2]) / n
        pies[:, n] = pi_n

        # Compute taus[:, n] out of place
        clone_of_pies = pies.clone()
        tau_n = (n + 1) * mu * clone_of_pies[:, n] - (n + 2) * clone_of_pies[:, n - 1]
        taus[:, n] = tau_n

    return pies, taus



def smat(k0, theta, r_c, eps_c, r_s=None, eps_s=None, eps_env=1.0, n_max=None):
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
    a_n = coreshell.an(x, y, n, m_c, m_s)
    b_n = coreshell.bn(x, y, n, m_c, m_s)

    print("a shape:", a_n.shape)
    print("b shape:", b_n.shape)

    print(a_n)

    print(b_n)

    mu = torch.cos(theta)

    pi, tau = pi_tau(n_max - 1, mu)

    print("pi shape:", pi.shape)
    print("tau shape:", tau.shape)

    print(pi)

    print(tau)

    n =+1

    s1 =  torch.sum( ((2 * n + 1)/(n*(n+1)))* (a_n*pi + b_n*tau), dim=1)
    s2 =  torch.sum( ((2 * n + 1)/(n*(n+1)))* (a_n*tau + b_n*pi), dim=1)

    I_sca = 0.5 * ( s1.real**2 + s1.imag**2 + s2.real**2 + s2.imag**2)

    return dict(
        S1 = s1,
        S2 = s2,
        I_sca = I_sca,
    )




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pymiediff
    torch.autograd.set_detect_anomaly(True)
    # z resolution
    N_pt_test = 10
    # n
    # n = torch.tensor(1)


    # theta.requires_grad = True

    # print(theta)

    # pi_tau_check = torch.autograd.gradcheck(pi_tau, (n, torch.cos(theta)), eps=0.01)


    # print(pi_tau_check)
    import pymiediff as pmd

    N_pt_test = 100
    N_order_test = 4

    dtype = torch.complex128  # torch.complex64
    device = torch.device("cpu")

    k0 = 2 * torch.pi / 200 #torch.linspace(200, 600, N_pt_test)
    theta = torch.linspace(0.01, 2*torch.pi -0.01, N_pt_test, dtype=torch.double)
    r_c = torch.tensor(12.0)
    r_s = torch.tensor(50.0)
    n_c = torch.tensor(2.0 + 0.2j)
    n_s = torch.tensor(5.0 + 0.1j)


    angular_scattering = pmd.angular.smat(
        k0=k0,
        theta=theta,
        r_c=r_c,
        eps_c=n_c**2,
        r_s=r_s,
        eps_s=n_s**2,
        eps_env=1,
        n_max=N_order_test,
    )

    s1 = angular_scattering["S1"]
    s2 = angular_scattering["S2"]
    angular_intensity = angular_scattering["I_sca"]
    print(s1)
    print(s2)
    print(angular_intensity)

    plt.plot(theta.detach().numpy(), angular_intensity.detach().numpy())
    plt.show()

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, constrained_layout=True)

    ax[0].plot(theta.detach().numpy(), s1.detach().numpy())
    ax[1].plot(theta.detach().numpy(), s2.detach().numpy())
    plt.show()





