# -*- coding: utf-8 -*-
"""
auto-diff ready wrapper of scipy spherical Bessel functions
"""
# %%
import warnings
from typing import Union

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
def sph_jn(n: torch.Tensor, z: torch.Tensor):
    """spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    _z = torch.as_tensor(z)
    _z = torch.atleast_1d(z)
    if _z.dim() == 1:
        _z = _z.unsqueeze(-1)
    
    result = _AutoDiffJn.apply(n, _z)
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
def sph_jn_der(n: torch.Tensor, z: torch.Tensor):
    """derivative of spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    _z = torch.as_tensor(z)
    _z = torch.atleast_1d(z)
    if _z.dim() == 1:
        _z = _z.unsqueeze(-1)
    
    result = _AutoDiffdJn.apply(n, _z)
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
def sph_yn(n: torch.Tensor, z: torch.Tensor):
    """spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    _z = torch.as_tensor(z)
    _z = torch.atleast_1d(z)
    if _z.dim() == 1:
        _z = _z.unsqueeze(-1)
    
    result = _AutoDiffYn.apply(n, _z)
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
def sph_yn_der(n: torch.Tensor, z: torch.Tensor):
    """derivative of spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    _z = torch.as_tensor(z)
    _z = torch.atleast_1d(z)
    if _z.dim() == 1:
        _z = _z.unsqueeze(-1)
    
    result = _AutoDiffdYn.apply(n, _z)
    return result


def sph_h1n(z: torch.Tensor, n: torch.Tensor):
    """spherical Hankel function of first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return sph_jn(n, z) + 1j * sph_yn(n, z)


def sph_h1n_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of spherical Hankel function of first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return sph_jn_der(n, z) + 1j * sph_yn_der(n, z)


# derived functions required for Mie
def psi(z: torch.Tensor, n: torch.Tensor):
    """Riccati-Bessel Function of the first kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return z * sph_jn(n, z)


def chi(z: torch.Tensor, n: torch.Tensor):
    """Riccati-Bessel Function of the secound kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return -z * sph_yn(n, z)


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
    return sph_jn(n, z) + z * sph_jn_der(n, z)


def chi_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of  Riccati-Bessel Function of the secound kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return -sph_yn(n, z) - z * sph_yn_der(n, z)


def xi_der(z: torch.Tensor, n: torch.Tensor):
    """derivative of  Riccati-Bessel Function of the third kind

    Args:
        z (torch.Tensor): complex argument
        n (torch.Tensor): integer order

    Returns:
        torch.Tensor: result
    """
    return sph_h1n(z, n) + z * sph_h1n_der(z, n)


# --- torch-native spherical Bessel functions via recurrences
def sph_jn_torch_via_rec(
    n: torch.Tensor, z: torch.Tensor, n_add="auto", eps=1e-7, **kwargs
):
    """Vectorized spherical Bessel of the first kind via downward recurrence

    Faster than continued-fraction ratios, but less stable for large |z| and medium large Im(z).
    Returns all orders. Vectorized over all z.
    Caution: May be unstable for medium and large |Im z|. Use continued-fraction ratios instead.

    Args:
        n (torch.Tensor or int): integer order(s)
        z (torch.Tensor): complex (or real) arguments to evalute
        n_add (str or int): 'auto' or integer extra depth for the downward recurrence.
                           'auto' picks a default based on max|z|. defaults to "auto"
        eps (float): minimum value for |z| to avoid numerical instability
        kwargs: other kwargs are ignored

    Returns:
        torch.Tensor: tensor of same shape of input z + (n_max+1,) dimension, where last dim indexes the order n=0..n_max.

    """
    n_max = int(n.max())
    assert n_max >= 0

    # some empirical automatic starting order guess
    if n_add.lower() == "auto":
        n_add = int(torch.abs(z).detach().cpu().max() * 2)
        if n_add > 30:
            n_add = 30

        n_add -= int(torch.angle(z).detach().cpu().abs().max() * 15)
        if n_add < 5:
            n_add = 5

    print("z: abs(ka)=", torch.max(torch.abs(z)), n_max, n_add)
    print("z: angle(ka)=", torch.max(torch.angle(z)), n_max, n_add)

    # ensure z is tensorial for broadcasting capability
    _z = z.clone()
    _z = torch.where(_z.abs() < eps, eps * torch.ones_like(_z), _z)
    _z = torch.atleast_1d(_z)
    if _z.dim() == 1:
        _z = _z.unsqueeze(-1)

    # allocate tensors
    jns = []  # use python list for Bessel orders to avoid in-place modif.

    j_n = torch.ones_like(_z) * 0.0
    j_np1 = torch.ones_like(_z) * 1e-25
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


def sph_jn_torch(
    n: Union[torch.Tensor, int],
    z: torch.Tensor,
    n_add: Union[str, int] = "auto",
    max_n_add: int = 50,
    small_z: float = 1e-8,
    precision="double",
    **kwargs,
):
    """Vectorized spherical Bessel of the first kind via continued-fraction ratios.

    Returns all orders.
    Small z are evaluated with Taylor series for more stability and efficiency.
    Vectorized over all z (we flatten then reshape back).
    Caution: May be unstable for extremely large |Im z|. You may try to increase n_add.

    Args:
        n (torch.Tensor or int): integer order(s)
        z (torch.Tensor): complex (or real) arguments to evalute
        n_add (str or int): 'auto' or integer extra depth for the continued fraction (seed truncation).
                           'auto' picks a safe default based on max|z|. defaults to "auto"
        max_n_add (int): upper bound for automatic extra depth (protects against huge loops). defaults to 50.
        small_z (float): threshold to treat z as small and use Taylor-series for those entries. Defaults to 1e-8.
        precision (str): "single" our "double". defaults to "double".
        kwargs: other kwargs are ignored

    Returns:
        torch.Tensor: tensor of same shape of input z + (n_max+1,) dimension, where last dim indexes order n=0..n_max.

    """
    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)

    # dtypes
    if precision.lower() == "single":
        dtype_f = torch.float32
        dtype_c = torch.complex64
    else:
        dtype_f = torch.float64
        dtype_c = torch.complex128

    # convert z to complex dtype and flatten
    z = torch.as_tensor(z).to(dtype_c)
    orig_shape = z.shape
    if z.dim() == 3:
        orig_shape = orig_shape[:-1]  # last axis of z is for order
    z_flat = z.reshape(-1)

    # prepare output (flattened)
    jns_flat = torch.zeros((z_flat.shape[0], n_max + 1), dtype=dtype_c, device=z.device)

    # Precompute orders array
    orders = torch.arange(0, n_max + 1, dtype=torch.int, device=z.device)

    # -- small z -- use series j_n(z) ~ z^n / (2n+1)!!  to avoid CF/div-by-zero issues
    abs_z_flat = torch.abs(z_flat)
    mask_small = abs_z_flat < small_z
    if mask_small.any():
        idx_small = mask_small.nonzero(as_tuple=True)[0]
        z_small = z_flat[idx_small]  # shape (S_small,)

        # double-factorial: log((2n+1)!!) = lgamma(2n+2) - n*log(2) - lgamma(n+1)
        n_arr = orders.to(torch.float64)
        log_dd = (
            torch.lgamma(2.0 * n_arr + 2.0)
            - n_arr * torch.log(torch.tensor(2.0))
            - torch.lgamma(n_arr + 1.0)
        )
        dd = torch.exp(log_dd).to(dtype=dtype_f)  # float -> cast when dividing below

        # leading-term series (very small z)
        zpow = z_small.unsqueeze(1) ** orders.to(dtype_c)  # shape (S_small, n_max+1)
        dd_complex = dd.to(dtype_c)
        jn_small = zpow / dd_complex.unsqueeze(0)

        # for n==0, j0 = 1 (z^0 / 1 ), correct; for exact z==0, higher orders are zero
        jns_flat[idx_small, :] = jn_small

    # -- large z -- continued-fraction for ratios
    idx_big = (~mask_small).nonzero(as_tuple=True)[0]
    if idx_big.numel() > 0:
        z_big = z_flat[idx_big]  # shape (Q,)
        n_z_big = z_big.shape[0]

        # - choose truncation depth (N = n_max + n_extra)
        if n_add == "auto":
            # use a conservative auto rule: more depth for bigger |z|
            zabs_max = float(torch.max(torch.abs(z_big)).item())
            n_extra_auto = max(20, int(zabs_max * 2.0) + 10)
            n_extra_use = min(n_extra_auto, max_n_add)
        else:
            n_extra_use = int(n_add)
        N = n_max + n_extra_use
        if N <= 0:
            N = n_max + 50

        # - allocate tensors
        r_storage = torch.zeros((n_z_big, n_max), dtype=dtype_c, device=z.device)
        r_next = torch.zeros((n_z_big,), dtype=dtype_c, device=z.device)
        jns_big = torch.zeros((n_z_big, n_max + 1), dtype=dtype_c, device=z.device)
        eps = torch.tensor(1e-300, dtype=dtype_c, device=z.device)  # avoid 1/0

        # - downward iteration: seed r_{N+1} = 0 (works if N large enough)
        # iterate from k = N down to 1
        # note: loop length ~ N (scalar loop) but operations inside are vectorized across Q
        for k in range(N, 0, -1):
            Ak = (2.0 * k + 1.0) / z_big  # shape (Q,)
            denom = Ak - r_next
            # avoid exact zeros in denominator
            denom = torch.where(denom.abs() == 0.0, denom + eps, denom)
            r_k = 1.0 / denom
            if k <= n_max:
                # store at index k-1 (r_1 .. r_n_max)
                r_storage[:, k - 1] = r_k
            r_next = r_k

        # - reconstruct j_n from j0 and cumulative product of ratios
        j0_big = torch.where(
            z_big == 0.0, torch.ones_like(z_big), torch.sin(z_big) / z_big
        )  # compute j0 safely

        # cumulative product of r_k across orders: shape (Q, n_max)
        r_cum = torch.cumprod(r_storage, dim=1)  # r_1, r_1*r_2, ...

        # fill spherical bessel output
        jns_big[:, 0] = j0_big
        if n_max >= 1:
            jns_big[:, 1:] = j0_big.unsqueeze(1) * r_cum
        jns_flat[idx_big, :] = jns_big

    # reshape jns_flat into original z shape plus orders axis
    jns = jns_flat.reshape(*orig_shape, n_max + 1)

    # for z == 0, enforce exact known limits
    mask0 = z == 0
    if mask0.any():
        # j_0(0)=1, j_n(0)=0 for n>0
        jns[mask0, 0] = 1.0
        jns[mask0, 1:] = 0.0

    return jns


def sph_yn_torch(n: torch.Tensor, z: torch.Tensor, eps=1e-7, **kwargs):
    """Vectorized spherical Bessel of the first kind via updward recurrence

    Returns all orders. Vectorized over all z.

    Args:
        n (torch.Tensor or int): integer order(s)
        z (torch.Tensor): complex (or real) arguments to evalute
        eps (float): minimum value for |z| to avoid numerical instability
        kwargs: other kwargs are ignored

    Returns:
        torch.Tensor: tensor of same shape of input z + (n_max+1,) dimension, where last dim indexes the order n=0..n_max.

    """
    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)
    assert n_max >= 0

    # ensure z is tensorial for broadcasting capability
    _z = z.clone()
    _z = torch.where(_z.abs() < eps, eps * torch.ones_like(_z), _z)
    _z = torch.atleast_1d(_z)
    if _z.dim() == 1:
        _z = _z.unsqueeze(-1)

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


def f_der_torch(n: torch.Tensor, z: torch.Tensor, f_n: torch.Tensor, **kwargs):
    """eval. derivatives of a spherical Bessel function (any unmodified)

    last axis of `z` and `f_n` is Mie order!

    use max of `n` as maximum order, last dimension of `f_n` contain the spherical bessel
    values at `z` and needs to carry all orders up to n.

    d/dz f_0 = -f_n+1 + (n/z) f_n, for n=0
    d/dz f_n = f_n-1 - (n+1)/z f_n, for n>0

    f_n: torch.Tensor of at least n=2

    Args:
        n (torch.Tensor or int): integer order(s)
        z (torch.Tensor): complex (or real) arguments to evalute
        f_n (torch.Tensor): Tensor containing f_n(z) for all z, where f_n is
            any unmodified spherical Bessel function (same shape as z).
        kwargs: other kwargs are ignored

    Returns:
        torch.Tensor: tensor of same shape as f_n

    """
    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)
    assert n_max >= 0

    f_n = torch.atleast_1d(f_n)
    if z.dim() < f_n.dim():
        z = torch.atleast_1d(z).unsqueeze(-1)  # add order dimension
    n_list = torch.arange(n_max + 1, device=z.device).broadcast_to(f_n.shape)

    df = torch.zeros_like(f_n)

    df[..., 0] = -f_n[..., 1]
    df[..., 1:] = f_n[..., :-1] - ((n_list[..., 1:] + 1) / z) * f_n[..., 1:]
    return df


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

    Jn_check = torch.autograd.gradcheck(sph_jn, (n, z1), eps=0.01)
    dJn_check = torch.autograd.gradcheck(sph_jn_der, (n, z2), eps=0.01)

    Yn_check = torch.autograd.gradcheck(sph_yn, (n, z3), eps=0.01)
    dYn_check = torch.autograd.gradcheck(sph_yn_der, (n, z4), eps=0.01)

    torch.autograd.set_detect_anomaly(True)

    pymiediff.helper.funct_grad_checker(
        z1, sph_jn, (n, z1), ax=(ax[0, 0], ax[0, 1]), check=Jn_check
    )
    pymiediff.helper.funct_grad_checker(
        z2, sph_jn_der, (n, z2), ax=(ax[1, 0], ax[1, 1]), check=dJn_check
    )
    pymiediff.helper.funct_grad_checker(
        z3, sph_yn, (n, z3), ax=(ax[2, 0], ax[2, 1]), check=Yn_check
    )
    pymiediff.helper.funct_grad_checker(
        z4, sph_yn_der, (n, z4), ax=(ax[3, 0], ax[3, 1]), check=dYn_check
    )

    plt.show()
