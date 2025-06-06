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


class _AutoDiffJn(torch.autograd.Function):
    @staticmethod
    def forward(n, z):
        n_np, z_np = n.detach().numpy(), z.detach().numpy()
        result = torch.from_numpy(spherical_jn(n_np, z_np))
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        n, z = inputs
        result = output
        ctx.save_for_backward(n, z, result)

    @staticmethod
    @torch.autograd.function.once_differentiable  # todo: double diff support
    def backward(ctx, grad_result):
        n, z, result = ctx.saved_tensors
        n_np, z_np = n.detach().numpy(), z.detach().numpy()

        # gradient of forward pass
        dz = torch.from_numpy(spherical_jn(n_np, z_np, derivative=True))

        # apply chain rule
        # torch convention: use conjugate!
        # see: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
        grad_wrt_z = grad_result * dz.conj()

        # differentiation wrt order `n` (int) is not allowed
        grad_wrt_n = None

        # return a gradient tensor for each input of "forward" (n, z)
        return grad_wrt_n, grad_wrt_z


# public API
def Jn(n: torch.Tensor, z: torch.Tensor):
    """spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffJn.apply(n, z)
    return result


class _AutoDiffdJn(torch.autograd.Function):
    @staticmethod
    def forward(n, z):
        n_np, z_np = n.detach().numpy(), z.detach().numpy()
        result = torch.from_numpy(spherical_jn(n_np, z_np, derivative=True))
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        n, z = inputs
        result = output
        ctx.save_for_backward(n, z, result)

    @staticmethod
    @torch.autograd.function.once_differentiable  # todo: double diff support
    def backward(ctx, grad_result):
        n, z, result = ctx.saved_tensors
        n_np, z_np = n.detach().numpy(), z.detach().numpy()

        # gradient of forward pass
        ddz = torch.from_numpy(bessel2ndDer(n_np, z_np, spherical_jn))

        # apply chain rule
        # torch convention: use conjugate!
        # see: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
        grad_wrt_z = grad_result * ddz.conj()

        # differentiation wrt order `n` (int) is not allowed
        grad_wrt_n = None

        # return a gradient tensor for each input of "forward" (n, z)
        return grad_wrt_n, grad_wrt_z


# public API
def dJn(n: torch.Tensor, z: torch.Tensor):
    """derivative of spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffdJn.apply(n, z)
    return result


class _AutoDiffYn(torch.autograd.Function):
    @staticmethod
    def forward(n, z):
        n_np, z_np = n.detach().numpy(), z.detach().numpy()
        result = torch.from_numpy(spherical_yn(n_np, z_np))
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        n, z = inputs
        result = output
        ctx.save_for_backward(n, z, result)

    @staticmethod
    @torch.autograd.function.once_differentiable  # todo: double diff support
    def backward(ctx, grad_result):
        n, z, result = ctx.saved_tensors
        n_np, z_np = n.detach().numpy(), z.detach().numpy()

        # gradient of forward pass
        dz = torch.from_numpy(spherical_yn(n_np, z_np, derivative=True))

        # apply chain rule
        # torch convention: use conjugate!
        # see: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
        grad_wrt_z = grad_result * dz.conj()

        # differentiation wrt order `n` (int) is not allowed
        grad_wrt_n = None

        # return a gradient tensor for each input of "forward" (n, z)
        return grad_wrt_n, grad_wrt_z


# public API
def Yn(n: torch.Tensor, z: torch.Tensor):
    """spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffYn.apply(n, z)
    return result


class _AutoDiffdYn(torch.autograd.Function):
    @staticmethod
    def forward(n, z):
        n_np, z_np = n.detach().numpy(), z.detach().numpy()
        result = torch.from_numpy(spherical_yn(n_np, z_np, derivative=True))
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        n, z = inputs
        result = output
        ctx.save_for_backward(n, z, result)

    @staticmethod
    @torch.autograd.function.once_differentiable  # todo: double diff support
    def backward(ctx, grad_result):
        n, z, result = ctx.saved_tensors
        n_np, z_np = n.detach().numpy(), z.detach().numpy()

        # gradient of forward pass
        ddz = torch.from_numpy(bessel2ndDer(n_np, z_np, spherical_yn))

        # apply chain rule
        # torch convention: use conjugate!
        # see: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers
        grad_wrt_z = grad_result * ddz.conj()

        # differentiation wrt order `n` (int) is not allowed
        grad_wrt_n = None

        # return a gradient tensor for each input of "forward" (n, z)
        return grad_wrt_n, grad_wrt_z


# public API
def dYn(n: torch.Tensor, z: torch.Tensor):
    """derivative of spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffdYn.apply(n, z)
    return result


def sph_h1n(z: torch.Tensor, n: torch.Tensor):
    """spherical Hankel function of first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return Jn(n, z) + 1j * Yn(n, z)


def sph_h1n_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of spherical Hankel function of first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return dJn(n, z) + 1j * dYn(n, z)


# torch-native via recurrences
## TODO: upward recurrence for n <= abs(x)/2 (then downward is unstable!!)
## TODO: chose n_add "on demand"
def sph_jn_torch(n: torch.Tensor, z: torch.Tensor, n_add=10, eps=1e-7):
    """via downward recurrence

    last axis is Mie order!

    returns a tensor of shape like `z` plus an additional, last
    dimension containing all evaluated orders

    eps: added to small values of `z` to avoid numerical instability

    returns all orders (0,...,n_max)

    """
    n_max = int(n.max())
    assert n_max >= 0

    # ensure z is tensorial for broadcasting capability
    _z = z.clone()
    _z = torch.where(_z.abs() < eps, eps * torch.ones_like(_z), _z)
    _z = torch.atleast_1d(_z)
    if _z.dim() == 1:
        _z.unsqueeze(-1)

    # allocate tensors
    jns = []  # use python list for Bessel orders to avoid in-place modif.

    j_n = torch.ones_like(_z)
    j_np1 = torch.ones_like(_z)
    j_nm1 = torch.zeros_like(_z)

    for _n in range(n_max + n_add, 0, -1):
        j_nm1 = ((2.0 * _n + 1.0) / _z) * j_n - j_np1
        j_np1 = j_n
        j_n = j_nm1
        if _n <= n_max + 1:
            jns.append(j_n[..., -1])

    # inverse order and convert to tensor
    jns = torch.stack(jns[::-1], dim=-1)  # last dim: order n

    # normalize
    j0_exact = torch.sin(_z[..., -1]) / _z[..., -1]
    scale = j0_exact / jns[..., 0]
    jns = jns * scale.unsqueeze(-1)

    return jns


def sph_yn_torch(n: torch.Tensor, z: torch.Tensor, eps=1e-7):
    """via upward recurrence

    last axis is Mie order!

    returns a tensor of shape like `z` plus an additional, last
    dimension containing all evaluated orders

    eps: added to small values of `z` to avoid numerical instability

    returns all orders (0,...,n_max)
    """
    n_max = int(n.max())
    assert n_max >= 0

    # ensure z is tensorial for broadcasting capability
    _z = z.clone()
    _z = torch.where(_z.abs() < eps, eps * torch.ones_like(_z), _z)
    _z = torch.atleast_1d(_z)
    if _z.dim() == 1:
        _z.unsqueeze(-1)

    # allocate tensors
    yns = []  # use python list for Bessel orders to avoid in-place modif.

    # zero order
    yns.append(-1 * (torch.cos(_z[..., -1]) / _z[..., -1]))

    # first order
    if n_max > 0:
        yns.append(
            -1
            * (
                (torch.cos(_z[..., -1]) / _z[..., -1] ** 2)
                + (torch.sin(_z[..., -1]) / _z[..., -1])
            )
        )

    # recurrence for higher orders
    if n_max > 1:
        for n_iter in range(2, n_max + 1):
            yns.append(
                ((2 * n_iter - 1) / _z[..., -1]) * (yns[n_iter - 1]) - yns[n_iter - 2]
            )

    # convert to tensor
    yns = torch.stack(yns, dim=-1)  # last dim: order n

    return yns


def f_prime_torch(n: torch.Tensor, z: torch.Tensor, f_n: torch.Tensor):
    """eval. derivative of a spherical Bessel function (any unmodified)

    last axis of `z` and `f_n` is Mie order!

    use max of `n` as maximum order, last dimension of `f_n` contain the spherical bessel
    values at `z` and needs to carry all orders up to n.

    d/dz f_0 = -f_n+1 + (n/z) f_n, for n=0
    d/dz d_n = f_n-1 - (n+1)/z f_n, for n>0

    f_n: torch.Tensor of at least n=2
    """
    n_max = int(n.max())
    assert n_max >= 0

    f_n = torch.atleast_1d(f_n)
    z = torch.atleast_1d(z)
    n_list = torch.arange(n_max + 1, device=z.device).broadcast_to(f_n.shape)

    df = torch.zeros_like(f_n)

    df[..., 0] = -f_n[..., 1]
    df[..., 1:] = f_n[..., :-1] - ((n_list[..., 1:] + 1) / z) * f_n[..., 1:]
    return df


# derived functions required for Mie
def psi(z: torch.Tensor, n: torch.Tensor):
    """Riccati-Bessel Function of the first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return z * Jn(n, z)


def chi(z: torch.Tensor, n: torch.Tensor):
    """Riccati-Bessel Function of the secound kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return -z * Yn(n, z)


def xi(z: torch.Tensor, n: torch.Tensor):
    """Riccati-Bessel Function of the third kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return z * sph_h1n(z, n)


def psi_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of Riccati-Bessel Function of the first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return Jn(n, z) + z * dJn(n, z)


def chi_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of  Riccati-Bessel Function of the secound kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return -Yn(n, z) - z * dYn(n, z)


def xi_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of  Riccati-Bessel Function of the third kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return sph_h1n(z, n) + z * sph_h1n_der(z, n)


# angular functions
def pi_tau(N: int, mu: torch.Tensor):
    """the angular functions tau and pi calculated by recurrence relation

    Args:
        N (int): integer order
        mu (torch.Tensor): cosine of the angle

    Returns:
        turple: turple of both results (pi and tua)
    """
    # Ensure N is an integer
    N = int(N)

    # Ensure mu is 1D to avoid shape mismatches
    mu = mu.view(-1)

    # Preallocate tensors for pi and tau with the correct shape
    pies = torch.zeros(len(mu), N + 1, dtype=mu.dtype, device=mu.device)
    taus = torch.zeros(len(mu), N + 1, dtype=mu.dtype, device=mu.device)

    # Initialize the first two terms
    pies[:, 0] = 1.0  # pi_0 = 1
    taus[:, 0] = mu  # tau_0 = mu
    if N > 0:
        pies[:, 1] = 3 * mu  # pi_1 = 3 * mu
        taus[:, 1] = 3 * torch.cos(2 * torch.acos(mu))  # tau_1 = 3cos(2acos(mu))

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pymiediff

    # z resolution
    N_pt_test = 50
    # n
    n = torch.tensor(1)
    # Jn
    z1 = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z1.requires_grad = True
    # dJn
    z2 = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z2.requires_grad = True
    # Yn
    z3 = torch.linspace(1, 2, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z3.requires_grad = True
    # dYn
    z4 = torch.linspace(1, 2, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z4.requires_grad = True

    fig, ax = plt.subplots(4, 2, figsize=(16, 9), dpi=100, constrained_layout=True)

    Jn_check = torch.autograd.gradcheck(Jn, (n, z1), eps=0.01)
    dJn_check = torch.autograd.gradcheck(dJn, (n, z2), eps=0.01)

    Yn_check = torch.autograd.gradcheck(Yn, (n, z3), eps=0.01)
    dYn_check = torch.autograd.gradcheck(dYn, (n, z4), eps=0.01)

    torch.autograd.set_detect_anomaly(True)

    pymiediff.helper.funct_grad_checker(
        z1, Jn, (n, z1), ax=(ax[0, 0], ax[0, 1]), check=Jn_check
    )
    pymiediff.helper.funct_grad_checker(
        z2, dJn, (n, z2), ax=(ax[1, 0], ax[1, 1]), check=dJn_check
    )
    pymiediff.helper.funct_grad_checker(
        z3, Yn, (n, z3), ax=(ax[2, 0], ax[2, 1]), check=Yn_check
    )
    pymiediff.helper.funct_grad_checker(
        z4, dYn, (n, z4), ax=(ax[3, 0], ax[3, 1]), check=dYn_check
    )

    plt.show()
