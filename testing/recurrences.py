# -*- coding: utf-8 -*-
"""
auto-diff ready wrapper of scipy spherical Bessel functions



"""
# %%
import warnings
from functools import lru_cache

import torch
from scipy.special import spherical_jn, spherical_yn
import numpy as np


def bessel2ndDer(n: torch.Tensor, z: torch.Tensor, bessel):
    """returns the secound derivative of a given bessel function

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument
        bessel (function): function to find secound derivative of

    Returns:
        torch.Tensor: result
    """
    z[z == 0] = 1e-10
    z = np.nan_to_num(z, nan=1e-10)
    return (1 / z**2) * ((n**2 - n - z**2) * bessel(n, z) + 2 * z * bessel(n + 1, z))


def sph_jn(n: int, z: torch.Tensor, n_add=10):
    """via downward recurrence

    returns a tensor of shape like `z` plus an additional, last
    dimension containing all evaluated orders
    """
    n = int(n)
    assert n >= 0

    # ensure z is tensorial for broadcasting capability
    z = torch.atleast_1d(z)

    # allocate tensors
    jns = torch.zeros(*z.shape, n + 1, dtype=z.dtype, device=z.device)

    j_n = torch.ones_like(z)
    j_np1 = torch.ones_like(z)
    j_nm1 = torch.zeros_like(z)

    for _n in range(n + n_add, 0, -1):
        j_nm1 = ((2.0 * _n + 1.0) / z) * j_n - j_np1
        j_np1 = j_n
        j_n = j_nm1
        if _n <= n + 1:
            jns[..., _n - 1] = j_n

    # normalize
    jns[..., 0] = torch.sin(z) / z
    if n >= 1:
        jns_clone = jns.clone() # added clone
        jns[..., 1:] = jns_clone[..., 1:] * (jns_clone[..., 0] / j_n).unsqueeze(-1)

    return jns


def sph_yn(n: int, z: torch.Tensor):
    """via upward recurrence

    returns a tensor of shape like `z` plus an additional, last
    dimension containing all evaluated orders
    """
    n = int(n)
    assert n >= 0

    # ensure z is tensorial for broadcasting capability
    z = torch.atleast_1d(z)

    # allocate tensors
    yns = torch.zeros(*z.shape, n + 1, dtype=z.dtype, device=z.device)

    yns[..., 0] = -1 * torch.cos(z) / z

    if n > 0:
        yns[..., 1] = -1 * torch.cos(z) / z**2 - torch.sin(z) / z

    if n > 1:
        for n in range(2, n + 1):
            yns[..., n] = ((2 * n - 1) / z) * yns[..., n - 1] - yns[..., n - 2]
    return yns


def f_prime(n: int, z: torch.Tensor, f_n: torch.Tensor):
    """eval. derivative of a spherical Bessel function (any unmodified)

    `n` is maximum order, las dimension of `f_n` contain the spherical bessel
    values at `z` and needs to carry all orders up to n.

    d/dz f_0 = -f_n+1 + (n/z) f_n, for n=0
    d/dz d_n = f_n-1 - (n+1)/z f_n, for n>0

    f_n: torch.Tensor of at least n=2
    """
    n = int(n)
    assert n >= 1

    f_n = torch.atleast_1d(f_n)
    z = torch.atleast_1d(z)
    n_list = torch.arange(n + 1).broadcast_to(f_n.shape)

    df = torch.zeros_like(f_n)

    df[..., 0] = -f_n[..., 1]
    df[..., 1:] = (
        f_n[..., :-1] - ((n_list[..., 1:] + 1) / z.unsqueeze(-1)) * f_n[..., 1:]
    )
    return df


def sph_djn(N: int, z: torch.Tensor):
    # Ensure integer
    N = int(N)
    # Ensure 1D
    z = z.view(-1)
    # Preallocate tensors
    jns = torch.zeros(len(z), N + 1, dtype=z.dtype, device=z.device)
    djns = torch.zeros(len(z), N + 1, dtype=z.dtype, device=z.device)

    jns[:, 0] = torch.sin(z) / z
    djns[:, 0] = (z * torch.cos(z) - torch.sin(z)) / z**2

    if N > 0:
        jns[:, 1] = torch.sin(z) / z - torch.cos(z) / z
        clone_of_jns = jns.clone()
        djns[:, 1] = clone_of_jns[:, 0] - (2 / z) * clone_of_jns[:, 1]
    for n in range(2, N + 1):
        # Compute pies[:, n] out of place
        clone_of_jns = jns.clone()
        j_n = ((2 * n + 1) / z) * clone_of_jns[:, n - 1] - clone_of_jns[:, n - 2]
        jns[:, n] = j_n
        clone_of_jns = jns.clone()
        dj_n = clone_of_jns[:, n - 1] - ((n + 1) / z) * clone_of_jns[:, n]
        djns[:, n] = dj_n
    return djns


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pymiediff as pmd

    # z resolution
    N_pt_test = 50
    # n
    n = torch.tensor(10)
    # Jn
    z1 = torch.linspace(5, 10, N_pt_test) + 0.1j * torch.linspace(0.5, 3, N_pt_test)
    z1 = torch.linspace(5, 10, N_pt_test * 2).reshape(2, N_pt_test)
    # z1 = torch.as_tensor([10.0])  # + 0.1j * torch.linspace(0.5, 3, N_pt_test)
    z1.requires_grad = True

    # first
    out = sph_jn(n, z1)
    out_prime = f_prime(n, z1, out)

    # grad test
    torch.autograd.set_detect_anomaly(True)
    grad = torch.autograd.grad(
        outputs=out, inputs=[z1], grad_outputs=torch.ones_like(out)
    )

    out_scipy = np.transpose([
        spherical_jn(_n, z1.detach().numpy()) for _n in range(n.detach().numpy() + 1)
    ])

    out_scipy_prime = np.transpose([
        spherical_jn(_n, z1.detach().numpy(), derivative=True) for _n in range(n.detach().numpy() + 1)
    ])

    # second
    out = sph_yn(n, z1)
    out_prime = f_prime(n, z1, out)
    out_scipy = np.transpose([
        spherical_yn(_n, z1.detach().numpy()) for _n in range(n.detach().numpy() + 1)
    ])

    out_scipy_prime = np.transpose([
        spherical_yn(_n, z1.detach().numpy(), derivative=True) for _n in range(n.detach().numpy() + 1)
    ])


    # - plot
    plot_t = out_prime
    plot_s = out_scipy_prime
    n_min_plot = 0
    idx0 = 1
    plt.plot(z1.detach().numpy()[idx0], plot_t.detach().numpy()[idx0, :, n_min_plot:])
    plt.gca().set_prop_cycle(None)  # reset color cycle
    plt.plot(
        z1.detach().numpy()[idx0], plot_s[:, idx0, n_min_plot:], lw=0, marker="x"
    )
    plt.show()

# %%
