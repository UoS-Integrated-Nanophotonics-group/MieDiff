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


def _expand_n_z(n, z, **kwargs):
    _z = torch.as_tensor(z, device=z.device)
    _z = torch.atleast_1d(z)
    # add order dimension (first dim.)
    _z = _z.unsqueeze(0)

    # order
    _n = torch.as_tensor(n, dtype=torch.int, device=z.device).squeeze()
    assert _n.nelement() == 1, "Mie order needs to be integer single element"
    _n_range = torch.arange(_n + 1, device=z.device)
    _n_range = _n_range[(...,) + (None,) * (_z.ndim - _n_range.ndim)]
    return _n_range, _z


def bessel2ndDer(n: torch.Tensor, z: torch.Tensor, bessel, **kwargs):
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
def sph_jn(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    _n, _z = _expand_n_z(n, z)

    result = _AutoDiffJn.apply(_n, _z)
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
def sph_jn_der(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """derivative of spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    _n, _z = _expand_n_z(n, z)

    result = _AutoDiffdJn.apply(_n, _z)
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
def sph_yn(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    _n, _z = _expand_n_z(n, z)

    result = _AutoDiffYn.apply(_n, _z)
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
def sph_yn_der(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """derivative of spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    _n, _z = _expand_n_z(n, z)

    result = _AutoDiffdYn.apply(_n, _z)
    return result


def sph_h1n(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """spherical Hankel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    # expanding n and z is done internally in `jn` and `yn`
    return sph_jn(n, z) + 1j * sph_yn(n, z)


def sph_h1n_der(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """derivative of spherical Hankel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    # expanding n and z is done internally in `jn` and `yn`
    return sph_jn_der(n, z) + 1j * sph_yn_der(n, z)


# generic derivatives
def f_der(n: torch.Tensor, z: torch.Tensor, f_n: torch.Tensor, **kwargs):
    """eval. derivatives of a spherical Bessel function (any unmodified)

    first axis of `z` and `f_n` must be Mie order!

    `n` is giving maximum order as integer, first dimension of `f_n` must also contain the order.
    `z` does not carry orders (all orders are evaluated at same `z`, therefore dim of z is 1 less than dim of `f_n`)

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
    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting

    df = torch.zeros_like(f_n)

    df[0, ...] = -f_n[1, ...]
    df[1:, ...] = f_n[:-1, ...] - ((_n[1:, ...] + 1) / _z) * f_n[1:, ...]
    return df


# Ricatti-Bessel via scipy API
def psi(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel Function of the first kind

    return Ricatti Bessel 1st kind as well as its derivatives

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument
        kwargs: additional kwargs are ignored

    Returns:
        torch.Tensor, torch.Tensor: direct result and derivative
    """
    # expanding n and z is done internally in `jn` and `yn`
    jn = sph_jn(n, z)
    jn_der = f_der(n, z, jn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    psi = _z * jn
    psi_der = jn + _z * jn_der

    return psi, psi_der


def chi(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel Function of the secound kind

    return Ricatti Bessel 2nd kind as well as its derivatives

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument
        kwargs: additional kwargs are ignored

    Returns:
        torch.Tensor: result
    """
    # expanding n and z is done internally in `jn` and `yn`
    yn = sph_yn(n, z)
    yn_der = f_der(n, z, yn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    chi = -_z * yn
    chi_der = -yn - _z * yn_der

    return chi, chi_der


def xi(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel Function of the third kind

    return Ricatti Bessel 3rd kind as well as its derivative

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument
        kwargs: additional kwargs are ignored

    Returns:
        torch.Tensor: result
    """
    # expanding n and z is done internally in `jn` and `yn`
    h1n = sph_h1n(n, z)
    h1n_der = f_der(n, z, h1n)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    xin = _z * h1n
    xin_der = h1n + _z * h1n_der

    return xin, xin_der


# --- torch-native spherical Bessel functions via recurrences
def sph_jn_torch_via_rec(
    n: torch.Tensor,
    z: torch.Tensor,
    n_add="auto",
    n_add_min=10,
    n_add_max=35,
    eps=1e-10,
    precision="double",
    **kwargs,
):
    """Vectorized spherical Bessel of the first kind via downward recurrence

    Seems a bit faster than continued-fraction ratios, but less stable for large |z| and medium large Im(z).
    Returns all orders. Vectorized over all z.
    Caution: May be unstable for medium and large |Im z|. Use continued-fraction ratios instead.

    Args:
        n (torch.Tensor or int): integer order(s)
        z (torch.Tensor): complex (or real) arguments to evalute
        n_add (str or int): 'auto' or integer extra depth for the downward recurrence.
                           'auto' picks a default based on max|z|. defaults to "auto"
        n_add_min (int): Minimum additional starting order. Defaults to 10.
        n_add_max (int): Maximum additional starting order. Defaults to 35.
        eps (float): minimum value for |z| to avoid numerical instability
        precision (str): "single" our "double". defaults to "double".
        kwargs: other kwargs are ignored

    Returns:
        torch.Tensor: tensor of same shape of input z + (n_max+1,) dimension, where last dim indexes the order n=0..n_max.

    """
    _n, _z = _expand_n_z(n, z)
    n_max = _n.max().item()
    assert n_max >= 0

    # dtypes
    if precision.lower() == "single":
        dtype_f = torch.float32
        dtype_c = torch.complex64
    else:
        dtype_f = torch.float64
        dtype_c = torch.complex128

    _z = _z.to(dtype=dtype_c)

    # add epsilon to small values for numerical stability
    _z = torch.where(_z.abs() < eps, eps * torch.ones_like(_z), _z)

    # some empirical automatic starting order guess
    if n_add.lower() == "auto":
        n_add = max(n_add_min, int(1.5 * torch.abs(z).detach().cpu().max()))
        if n_add > n_add_max:
            n_add = n_add_max

    # allocate tensors
    jns = []  # use python list for Bessel orders to avoid in-place modif.

    j_n = torch.ones_like(_z, dtype=dtype_c) * 0.0
    j_np1 = torch.ones_like(_z, dtype=dtype_c) * 1e-25
    j_nm1 = torch.zeros_like(_z, dtype=dtype_c)

    for _n in range(n_max + n_add, 0, -1):
        j_nm1 = ((2.0 * _n + 1.0) / _z) * j_n - j_np1
        j_np1 = j_n
        j_n = j_nm1
        if _n <= n_max + 1:
            jns.append(j_n[-1, ...])

    # inverse order and convert to tensor
    jns = torch.stack(jns[::-1], dim=0)  # first dim: order n

    # normalize
    j0_exact = torch.sin(_z[-1, ...]) / _z[-1, ...]
    scale = j0_exact / jns[0, ...]
    jns = jns * scale.unsqueeze(0)  # scale: expand order dim.

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
    _n, _z = _expand_n_z(n, z)
    n_max = _n.max().item()
    assert n_max >= 0

    # dtypes
    if precision.lower() == "single":
        dtype_f = torch.float32
        dtype_c = torch.complex64
    else:
        dtype_f = torch.float64
        dtype_c = torch.complex128

    # convert z to requested precision and flatten
    _z = _z.to(dtype=dtype_c)
    orig_shape = _z.shape
    if _z.dim() > 1:
        orig_shape = orig_shape[1:]  # first axis is order n
    z_flat = _z.reshape(-1)

    # prepare output (flattened)
    jns_flat = torch.zeros((n_max + 1, z_flat.shape[0]), dtype=dtype_c, device=z.device)

    # -- small z: use series j_n(z) ~ z^n / (2n+1)!!  to avoid CF/div-by-zero issues
    abs_z_flat = torch.abs(z_flat)
    mask_small = abs_z_flat < small_z
    if mask_small.any():
        idx_small = mask_small.nonzero(as_tuple=True)[0]
        z_small = z_flat[idx_small]  # shape (S_small,)

        # double-factorial: log((2n+1)!!) = lgamma(2n+2) - n*log(2) - lgamma(n+1)
        n_arr = _n.to(dtype_f)
        log_dd = (
            torch.lgamma(2.0 * n_arr + 2.0)
            - n_arr * torch.log(torch.tensor(2.0))
            - torch.lgamma(n_arr + 1.0)
        )
        dd = torch.exp(log_dd).to(dtype_f)

        # leading-term series (very small z)
        zpow = z_small.unsqueeze(0) ** _n.to(dtype_c)  # shape (n_max+1, S_small)
        jn_small = zpow / dd.to(dtype_c).unsqueeze(0)

        # for n==0, j0 = 1 (z^0 / 1 ), correct; for exact z==0, higher orders are zero
        jns_flat[:, idx_small] = jn_small

    # -- large z: continued-fraction for ratios
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

        # - allocate tensors
        r_storage = torch.zeros((n_max, n_z_big), dtype=dtype_c, device=z.device)
        r_next = torch.zeros((n_z_big,), dtype=dtype_c, device=z.device)
        jns_big = torch.zeros((n_max + 1, n_z_big), dtype=dtype_c, device=z.device)
        eps = torch.tensor(1e-50, dtype=dtype_c, device=z.device)  # avoid 1/0

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
                r_storage[k - 1, :] = r_k
            r_next = r_k

        # - reconstruct j_n from j0 and cumulative product of ratios
        j0_big = torch.where(
            z_big == 0.0, torch.ones_like(z_big), torch.sin(z_big) / z_big
        )  # compute j0 safely

        # cumulative product of r_k across orders: shape (Q, n_max)
        r_cum = torch.cumprod(r_storage, dim=0)  # r_1, r_1*r_2, ...

        # fill spherical bessel output
        jns_big[0, :] = j0_big
        if n_max >= 1:
            jns_big[1:, :] = j0_big.unsqueeze(0) * r_cum
        jns_flat[:, idx_big] = jns_big

    # reshape jns_flat into original z shape plus orders axis
    jns = jns_flat.reshape(n_max + 1, *orig_shape)

    # for z == 0, enforce exact known limits:
    # j_0(0)=1, j_n(0)=0 for n>0
    mask0 = _z[0] == 0
    if mask0.any():
        jns[0, mask0] = 1.0
        jns[1:, mask0] = 0.0

    return jns


def sph_yn_torch(n: torch.Tensor, z: torch.Tensor, eps=1e-10, **kwargs):
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
    _n, _z = _expand_n_z(n, z)
    n_max = _n.max().item()
    assert n_max >= 0

    # add epsilon to small values for numerical stability
    _z = torch.where(_z.abs() < eps, eps * torch.ones_like(_z), _z)

    # allocate tensors
    yns = []  # use python list for Bessel orders to avoid in-place modif.

    # zero order
    yns.append(-1 * (torch.cos(_z[-1, ...]) / _z[-1, ...]))

    # first order
    if n_max > 0:
        yns.append(
            -1
            * (
                (torch.cos(_z[-1, ...]) / _z[-1, ...] ** 2)
                + (torch.sin(_z[-1, ...]) / _z[-1, ...])
            )
        )

    # recurrence for higher orders
    if n_max > 1:
        for n_iter in range(2, n_max + 1):
            yns.append(
                ((2 * n_iter - 1) / _z[-1, ...]) * (yns[n_iter - 1]) - yns[n_iter - 2]
            )

    # convert to tensor
    yns = torch.stack(yns, dim=0)  # first dim: order n

    return yns


def sph_h1n_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """spherical Hankel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    return sph_jn_torch(n, z, **kwargs) + 1j * sph_yn_torch(n, z, **kwargs)


# Ricatti-Bessel
def psi_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel Function of the first kind

    return Ricatti Bessel as well as its derivative

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor, torch.Tensor: direct result and derivative
    """
    jn = sph_jn_torch(n, z, **kwargs)
    jn_der = f_der(n, z, jn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    psi = _z * jn
    psi_der = jn + _z * jn_der

    return psi, psi_der


def chi_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel Function of the secound kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    yn = sph_yn_torch(n, z, **kwargs)
    yn_der = f_der(n, z, yn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    chi = -_z * yn
    chi_der = -yn - _z * yn_der

    return chi, chi_der


def xi_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel Function of the third kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    h1n = sph_h1n_torch(n, z, **kwargs)
    h1n_der = f_der(n, z, h1n)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    xin = _z * h1n
    xin_der = h1n + _z * h1n_der

    return xin, xin_der


# angular functions
def pi_tau(n: int, mu: torch.Tensor, **kwargs):
    """angular functions tau and pi up to order n

    calculated by recurrence relation. Returns all orders up to n,
    add a new "order" dimension (first dim) to the results

    Uses recurrence:
        pi_0 = 0, pi_1 = 1
        pi_{n+1} = ((2n+1)*mu*pi_n - (n+1)*pi_{n-1}) / n
        tau_n = n*mu*pi_n - (n+1)*pi_{n-1}

    Args:
        n (torch.Tensor or int): order, use max(n) if a tensor.
        mu (torch.Tensor): cosine of the angle

    Returns:
        turple: tuple (pi, tau).
    """
    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)
    assert n_max >= 1

    # keep original shape to reshape later
    orig_shape = mu.shape
    mu_flat = mu.view(-1)  # 1D for iteration
    L = mu_flat.shape[0]

    # arrays indexed 0..n_max (we'll fill 0..n_max then return 1..n_max)
    pi_all = torch.zeros((n_max + 1, L), dtype=mu_flat.dtype, device=mu_flat.device)
    tau_all = torch.zeros_like(pi_all)

    # initial conditions (bhmie convention)
    pi_all[0, :] = 0.0  # pi_0 = 0 (not used in practice)
    pi_all[1, :] = 1.0  # pi_1 = 1
    tau_all[0, :] = 0.0  # tau_0 = 0 (not used in practice)
    tau_all[1, :] = mu_flat  # tau_1 = μ

    # upward recurrence for pi
    # compute pi_{n+1} from pi_n and pi_{n-1}
    for nn in range(1, n_max):
        # compute pi_{nn+1}
        pi_all[nn + 1, :] = (
            (2 * nn + 1) * mu_flat * pi_all[nn, :] - (nn + 1) * pi_all[nn - 1, :]
        ) / nn

    # compute tau for n = 1..n_max
    for nn in range(1, n_max + 1):
        tau_all[nn, :] = nn * mu_flat * pi_all[nn, :] - (nn + 1) * pi_all[nn - 1, :]

    # return shapes with order dim first and without the n=0 slot
    pi = pi_all.view((n_max + 1,) + orig_shape)
    tau = tau_all.view((n_max + 1,) + orig_shape)
    return pi, tau


def vsh(n_max, k0, n_medium, r, theta, phi, kind, epsilon=1e-8):
    """vector spherical harmonics for l=1

    Args:
        n_max (int): Maximum evaluation order (all up to this will be returned)
        k0 (torch.Tensor): vacuum wavenumber
        n_medium (torch.Tensor): refractive index of medium
        r (torch.Tensor): radial coordiantes
        theta (torch.Tensor): polar angle coordinates
        phi (torch.Tensor): azimuthal angle coordinates
        kind (int): radial dependence. 1: j_n, 2: y_n, 3: h1_n.
        epsilon (float, optional): small numerical value to avoid singularity at origin. Defaults to 1e-8.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    rho = k0 * n_medium * r + epsilon

    # angular function evaluation
    pi_n, tau_n = pi_tau(n_max, cos_t[0])  # pass `cos_t` without order dim. (dim 0)
    # Mie orders n = 1, ..., n_max (remove zero order)
    pi_n = pi_n[1:]
    tau_n = tau_n[1:]

    # - inner or outer fields (j_n or h1_n)
    if kind == 1:
        # using j_n
        rho_zn, rho_zn_der = psi_torch(n_max, rho[0])
    elif kind == 2:
        # using y_n
        rho_zn, rho_zn_der = chi_torch(n_max, rho[0])
    elif kind == 3:
        # using h1_n
        rho_zn, rho_zn_der = xi_torch(n_max, rho[0])
    else:
        raise ValueError("`kind` parameter must be either 1, 2 or 3.")

    # Mie orders n = 1, ..., n_max (remove zero order)
    rho_zn = rho_zn[1:]
    rho_zn_der = rho_zn_der[1:]
    rho_zn_der_over_rho = rho_zn_der / rho
    zn = rho_zn / rho

    # define spherical coord. unit vector convention (shape broadcastable to r, theta and phi)
    e_r = torch.as_tensor([1, 0, 0], device=k0.device).view((r.ndim - 1) * (1,) + (-1,))
    e_tet = torch.as_tensor([0, 1, 0], device=k0.device).view(
        (r.ndim - 1) * (1,) + (-1,)
    )
    e_phi = torch.as_tensor([0, 0, 1], device=k0.device).view(
        (r.ndim - 1) * (1,) + (-1,)
    )

    # all mie orders (broadcastable to spherical positions)
    n = torch.arange(1, n_max + 1, device=k0.device)
    n = n.view((-1,) + (r.ndim - 1) * (1,))

    # odd
    M_o1n = cos_p * pi_n * zn * e_tet - sin_p * tau_n * zn * e_phi
    N_o1n = (sin_p * n * (n + 1) * sin_t * pi_n * (zn / rho) * e_r) + (
        (sin_p * tau_n * rho_zn_der_over_rho * e_tet)
        + (cos_p * pi_n * rho_zn_der_over_rho * e_phi)
    )

    # even
    M_e1n = -sin_p * pi_n * zn * e_tet - cos_p * tau_n * zn * e_phi
    N_e1n = (cos_p * n * (n + 1) * sin_t * pi_n * (zn / rho) * e_r) + (
        (cos_p * tau_n * rho_zn_der_over_rho * e_tet)
        - (sin_p * pi_n * rho_zn_der_over_rho * e_phi)
    )

    return M_o1n, M_e1n, N_o1n, N_e1n


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
