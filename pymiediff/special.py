# -*- coding: utf-8 -*-
"""
auto-diff ready wrapper of scipy spherical Bessel functions
"""
# %%
import warnings

import torch
from scipy.special import spherical_jn, spherical_yn
import numpy as np

# import numpy as np


def bessel2ndDer(n, z, bessel):
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


def sph_h1n(z, n):
    return Jn(n, z) + 1j * Yn(n, z)


def sph_h1n_der(z, n):
    return dJn(n, z) + 1j * dYn(n, z)


def psi(z, n):
    return z * Jn(n, z)


def chi(z, n):
    return -z * Yn(n, z)


def xi(z, n):
    return z * sph_h1n(z, n)


def psi_der(z, n):
    return Jn(n, z) + z * dJn(n, z)


def chi_der(z, n):
    return -Yn(n, z) - z * dYn(n, z)


def xi_der(z, n):
    return sph_h1n(z, n) + z * sph_h1n_der(z, n)


def pi_n(N, theta):
    N = torch.as_tensor(N, dtype=torch.int)
    if N == 0:
        return torch.tensor(0.0, dtype=theta.dtype, device=theta.device)
    elif N == 1:
        return torch.tensor(1.0, dtype=theta.dtype, device=theta.device)
    pi_nm2 = torch.tensor(0.0, dtype=theta.dtype, device=theta.device)
    pi_nm1 = torch.tensor(1.0, dtype=theta.dtype, device=theta.device)

    for n in range(2, N + 1):
        pi_n = ((2 * n - 1) / (n - 1)) * torch.cos(theta) * pi_nm1 - (
            n / (n - 1)
        ) * pi_nm2
        pi_nm2 = pi_nm1
        pi_nm1 = pi_n
    return pi_n


def tau_n(N, theta):
    N = torch.as_tensor(N, dtype=torch.int)
    if N == 0:
        return torch.tensor(0.0, dtype=theta.dtype, device=theta.device)
    for n in range(1, N + 1):
        tau_n = n * torch.cos(theta) * pi_n(n, theta) - (n - 1) * pi_n(n - 1, theta)
    return tau_n


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pymiediff

    # z resolution
    N_pt_test = 50
    # n
    n = torch.tensor(5.0)
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
    #pi
    theta1 = torch.linspace(0.01, 2*torch.pi, N_pt_test)
    theta1.requires_grad = True
    #tau
    theta2 = torch.linspace(0.01, 2*torch.pi, N_pt_test)
    theta2.requires_grad = True


    fig, ax = plt.subplots(6, 2, figsize=(16, 9), dpi=100, constrained_layout=True)

    Jn_check = torch.autograd.gradcheck(Jn, (n, z1), eps=0.01)
    dJn_check = torch.autograd.gradcheck(dJn, (n, z2), eps=0.01)

    Yn_check = torch.autograd.gradcheck(Yn, (n, z3), eps=0.01)
    dYn_check = torch.autograd.gradcheck(dYn, (n, z4), eps=0.01)
    try:
        pi_check = torch.autograd.gradcheck(pi_n, (n, theta1), eps=0.01)
    except:
        pi_check = False
    try:
        tau_check = torch.autograd.gradcheck(tau_n, (n, theta2), eps=0.01)
    except:
        tau_check = False



    pymiediff.helper.FunctGradChecker(
        z1, Jn, (n, z1), ax=(ax[0, 0], ax[0, 1]), check=Jn_check
    )
    pymiediff.helper.FunctGradChecker(
        z2, dJn, (n, z2), ax=(ax[1, 0], ax[1, 1]), check=dJn_check
    )
    pymiediff.helper.FunctGradChecker(
        z3, Yn, (n, z3), ax=(ax[2, 0], ax[2, 1]), check=Yn_check
    )
    pymiediff.helper.FunctGradChecker(
        z4, dYn, (n, z4), ax=(ax[3, 0], ax[3, 1]), check=dYn_check
    )
    pymiediff.helper.FunctGradChecker(
        theta1, pi_n, (n, theta1), ax=(ax[4, 0], ax[4, 1]), imag=False, check=pi_check)
    pymiediff.helper.FunctGradChecker(
        theta2, tau_n, (n, theta2), ax=(ax[5, 0], ax[5, 1]), imag=False, check=tau_check)

    plt.show()

# %%
