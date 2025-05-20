# -*- coding: utf-8 -*-
"""
auto-diff ready wrapper of scipy spherical Bessel functions
"""
# %%
import warnings

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


def sph_jn(N: int, z: torch.Tensor, n_add=10):
    """via downward recurrence"""
    # Ensure integer
    N = int(N)
    assert N >= 0

    # Ensure 1D
    z = torch.atleast_1d(z)
    assert z.dim() == 1

    # Preallocate tensors
    jns = torch.zeros(len(z), N + 1, dtype=z.dtype, device=z.device)

    j_n = torch.ones_like(z)
    j_np1 = torch.ones_like(z)
    j_nm1 = torch.zeros_like(z)

    for _n in range(N + n_add, 0, -1):
        j_nm1 = ((2.0 * _n + 1.0) / z) * j_n - j_np1
        j_np1 = j_n
        j_n = j_nm1
        if _n <= N + 1:
            jns[:, _n - 1] = j_n

    # normalize
    jns[:, 0] = torch.sin(z) / z
    if N >= 1:
        jns[:, 1:] = jns[:, 1:] * (jns[:, 0] / j_n).unsqueeze(1)

    return jns


def sph_yn(N: int, z: torch.Tensor):
    """via upward recurrence"""
    # Ensure integer
    N = int(N)
    assert N >= 0

    # Ensure 1D
    z = torch.atleast_1d(z)
    assert z.dim() == 1

    # allocate tensors
    yns = torch.zeros(len(z), N + 1, dtype=z.dtype, device=z.device)

    yns[:, 0] = -1 * torch.cos(z) / z

    if N > 0:
        yns[:, 1] = -1 * torch.cos(z) / z**2 - torch.sin(z) / z

    if N > 1:
        for n in range(2, N + 1):
            yns[:, n] = ((2 * n - 1) / z) * yns[:, n - 1] - yns[:, n - 2]
    return yns


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
    # z1 = torch.as_tensor([10.0])  # + 0.1j * torch.linspace(0.5, 3, N_pt_test)
    z1.requires_grad = True

    out = sph_jn(n, z1)
    out_scipy = [
        spherical_jn(_n, z1.detach().numpy()) for _n in range(n.detach().numpy() + 1)
    ]
    out_scipy = np.transpose(out_scipy)

    out = sph_yn(n, z1)
    out_scipy = [
        spherical_yn(_n, z1.detach().numpy()) for _n in range(n.detach().numpy() + 1)
    ]
    out_scipy = np.transpose(out_scipy)

    n_min_plot = 0
    plt.plot(z1.detach().numpy(), out.detach().numpy()[:, n_min_plot:])
    plt.gca().set_prop_cycle(None)  # reset color cycle
    plt.plot(z1.detach().numpy(), out_scipy[:, n_min_plot:], lw=0, marker="x")
    plt.show()

# %%


