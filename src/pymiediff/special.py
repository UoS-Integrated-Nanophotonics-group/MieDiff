# -*- coding: utf-8 -*-
"""Special functions used by :mod:`pymiediff`.

This module provides autograd-friendly spherical Bessel/Hankel functions and
their derivatives, together with Riccati-Bessel forms and vector spherical
harmonic (VSH) helpers used by Mie coefficient and near-field solvers.

Both SciPy-backed and torch-native recurrence implementations are available.
The Peña/Yang log-derivative recurrences are also implemented for stable
multilayer computations.
"""
# %%
import warnings
from typing import Union

import torch
from scipy.special import spherical_jn, spherical_yn
import numpy as np


def _expand_n_z(n, z, **kwargs):
    """Broadcast order and argument tensors for order-wise recurrences.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Evaluation points.

    Returns
    -------
    tuple of torch.Tensor
        ``(_n_range, _z)`` where ``_n_range`` is ``0..n`` and ``_z`` has a
        leading singleton order axis.
    """
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

def _expand_n_z_l(n, z, l, **kwargs):
    return


def bessel2ndDer(n: torch.Tensor, z: torch.Tensor, bessel, **kwargs):
    """Evaluate second derivative of a spherical Bessel function.

    Parameters
    ----------
    n : torch.Tensor
        Integer order.
    z : torch.Tensor
        Complex argument.
    bessel : callable
        Function implementing ``bessel(n, z)``.

    Returns
    -------
    torch.Tensor
        Second derivative with respect to ``z``.
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
    """Spherical Bessel function of the first kind.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        Values ``j_n(z)`` for orders ``0..n``.
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
    """Derivative of spherical Bessel function of the first kind.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        Values ``d j_n(z) / dz`` for orders ``0..n``.
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
    """Spherical Bessel function of the second kind.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        Values ``y_n(z)`` for orders ``0..n``.
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
    """Derivative of spherical Bessel function of the second kind.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        Values ``d y_n(z) / dz`` for orders ``0..n``.
    """
    _n, _z = _expand_n_z(n, z)

    result = _AutoDiffdYn.apply(_n, _z)
    return result


def sph_h1n(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Spherical Hankel function of the first kind.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        Values ``h_n^{(1)}(z)`` for orders ``0..n``.
    """
    # expanding n and z is done internally in `jn` and `yn`
    return sph_jn(n, z) + 1j * sph_yn(n, z)


def sph_h1n_der(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Derivative of spherical Hankel function of the first kind.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        Values ``d h_n^{(1)}(z) / dz`` for orders ``0..n``.
    """
    # expanding n and z is done internally in `jn` and `yn`
    return sph_jn_der(n, z) + 1j * sph_yn_der(n, z)


# generic derivatives
def f_der(n: torch.Tensor, z: torch.Tensor, f_n: torch.Tensor, **kwargs):
    """Differentiate order-indexed spherical Bessel-like sequences.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Argument values.
    f_n : torch.Tensor
        Sequence values for orders ``0..n`` along axis 0.

    Returns
    -------
    torch.Tensor
        Derivative values with same shape as ``f_n``.
    """
    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting

    df = torch.zeros_like(f_n)

    df[0, ...] = -f_n[1, ...]
    df[1:, ...] = f_n[:-1, ...] - ((_n[1:, ...] + 1) / _z) * f_n[1:, ...]
    return df


# Ricatti-Bessel via scipy API
def psi(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel function ``psi_n(z)`` and derivative.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    tuple of torch.Tensor
        ``(psi_n, dpsi_n_dz)`` for orders ``0..n``.
    """
    # expanding n and z is done internally in `jn` and `yn`
    jn = sph_jn(n, z)
    jn_der = f_der(n, z, jn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    psi = _z * jn
    psi_der = jn + _z * jn_der

    return psi, psi_der


def chi(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel function ``chi_n(z)`` and derivative.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    tuple of torch.Tensor
        ``(chi_n, dchi_n_dz)`` for orders ``0..n``.
    """
    # expanding n and z is done internally in `jn` and `yn`
    yn = sph_yn(n, z)
    yn_der = f_der(n, z, yn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    chi = -_z * yn
    chi_der = -yn - _z * yn_der

    return chi, chi_der


def xi(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Riccati-Bessel function ``xi_n(z)`` and derivative.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    tuple of torch.Tensor
        ``(xi_n, dxi_n_dz)`` for orders ``0..n``.
    """
    # expanding n and z is done internally in `jn` and `yn`
    h1n = sph_h1n(n, z)
    h1n_der = f_der(n, z, h1n)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    xin = _z * h1n
    xin_der = h1n + _z * h1n_der

    return xin, xin_der


# --- Peña/Yang log-derivative recurrences (multilayer backend)
def _pena_nmax(n):
    """Convert order input to scalar Python ``int`` maximum order."""
    if isinstance(n, torch.Tensor):
        return int(torch.as_tensor(n).max().item())
    return int(n)


def pena_D1_n(
    n: Union[torch.Tensor, int],
    z: torch.Tensor,
    n_add: int = 15,
    eps: float = 1e-30,
    precision: str = "double",
):
    """Compute ``D_n^(1)(z) = psi'_n(z) / psi_n(z)`` by downward recurrence.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).
    n_add : int, default=15
        Extra recurrence depth for stable seeding.
    eps : float, default=1e-30
        Small denominator safeguard.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.

    Returns
    -------
    torch.Tensor
        Log-derivative values for orders ``0..n``.
    """
    n_max = _pena_nmax(n)
    if n_max < 0:
        raise ValueError("`n` must be >= 0.")

    _z = torch.atleast_1d(torch.as_tensor(z))
    if precision.lower() == "single":
        dtype_c = torch.complex64
    else:
        dtype_c = torch.complex128
    _z = _z.to(dtype=dtype_c)

    # build entries without in-place tensor writes (autograd-safe)
    D_list = [None] * (n_max + 1)

    n_start = n_max + int(n_add)
    D_next = torch.zeros_like(_z, dtype=dtype_c)
    z_safe = torch.where(_z.abs() < eps, _z + eps, _z)

    for nn in range(n_start, 0, -1):
        num = nn / z_safe
        den = D_next + num
        den = torch.where(den.abs() < eps, den + eps, den)
        D_curr = num - 1.0 / den  # this is D_{nn-1}
        if nn - 1 <= n_max:
            D_list[nn - 1] = D_curr
        D_next = D_curr

    return torch.stack(D_list, dim=0)


def pena_D3_n(
    n: Union[torch.Tensor, int],
    z: torch.Tensor,
    D1: torch.Tensor = None,
    eps: float = 1e-30,
    precision: str = "double",
):
    """Compute ``D_n^(3)(z) = zeta'_n(z) / zeta_n(z)`` recurrence.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).
    D1 : torch.Tensor, optional
        Precomputed ``D1`` values.
    eps : float, default=1e-30
        Small denominator safeguard.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.

    Returns
    -------
    torch.Tensor
        Log-derivative values for orders ``0..n``.
    """
    n_max = _pena_nmax(n)
    if n_max < 0:
        raise ValueError("`n` must be >= 0.")

    _z = torch.atleast_1d(torch.as_tensor(z))
    if precision.lower() == "single":
        dtype_c = torch.complex64
    else:
        dtype_c = torch.complex128
    _z = _z.to(dtype=dtype_c)

    if D1 is None:
        D1 = pena_D1_n(n_max, _z, precision=precision)
    D1 = D1.to(dtype=dtype_c)

    # psi0(z) * zeta0(z) = 0.5 * (1 - exp(2 i z))
    psi_zeta_prev = 0.5 * (1.0 - torch.exp(2j * _z))
    D3_prev = 1j * torch.ones_like(_z, dtype=dtype_c)
    D3_list = [D3_prev]

    z_safe = torch.where(_z.abs() < eps, _z + eps, _z)
    for nn in range(1, n_max + 1):
        t1 = (nn / z_safe) - D1[nn - 1, ...]
        t2 = (nn / z_safe) - D3_prev
        psi_zeta_curr = psi_zeta_prev * t1 * t2
        denom = torch.where(psi_zeta_curr.abs() < eps, psi_zeta_curr + eps, psi_zeta_curr)
        D3_curr = D1[nn, ...] + 1j / denom
        D3_list.append(D3_curr)
        psi_zeta_prev = psi_zeta_curr
        D3_prev = D3_curr

    return torch.stack(D3_list, dim=0)




def pena_Q_n(
    n: Union[torch.Tensor, int],
    z1: torch.Tensor,
    z2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    D1_z1: torch.Tensor,
    D1_z2: torch.Tensor,
    D3_z1: torch.Tensor,
    D3_z2: torch.Tensor,
    eps: float = 1e-30,
    precision: str = "double",
):
    """Evaluate Peña/Pal interface ratio recurrence ``Q_n``.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z1, z2, x1, x2 : torch.Tensor
        Interface recurrence arguments.
    D1_z1, D1_z2, D3_z1, D3_z2 : torch.Tensor
        Precomputed logarithmic derivatives.
    eps : float, default=1e-30
        Small denominator safeguard.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.

    Returns
    -------
    torch.Tensor
        Ratio sequence for orders ``0..n``.
    """
    n_max = _pena_nmax(n)
    _z1 = torch.atleast_1d(torch.as_tensor(z1))
    _z2 = torch.atleast_1d(torch.as_tensor(z2))
    _x1 = torch.atleast_1d(torch.as_tensor(x1))
    _x2 = torch.atleast_1d(torch.as_tensor(x2))

    if precision.lower() == "single":
        dtype_c = torch.complex64
    else:
        dtype_c = torch.complex128
    _z1 = _z1.to(dtype=dtype_c)
    _z2 = _z2.to(dtype=dtype_c)
    _x1 = _x1.to(dtype=dtype_c)
    _x2 = _x2.to(dtype=dtype_c)

    # Eq. (19a): Q_0^(l)
    a1 = _z1.real
    b1 = _z1.imag
    a2 = _z2.real
    b2 = _z2.imag
    num0 = torch.exp(-2j * a1) - torch.exp(-2.0 * b1)
    den0 = torch.exp(-2j * a2) - torch.exp(-2.0 * b2)
    den0 = torch.where(den0.abs() < eps, den0 + eps, den0)
    Q_prev = (num0 / den0) * torch.exp(-2.0 * (b2 - b1))
    Q_list = [Q_prev]

    # Eq. (19b): Q_n^(l), n >= 1
    x2_safe = torch.where(_x2.abs() < eps, _x2 + eps, _x2)
    x_ratio_sq = (_x1 / x2_safe) ** 2
    for nn in range(1, n_max + 1):
        n_c = torch.as_tensor(nn, dtype=dtype_c, device=_z1.device)
        num = (_z2 * D1_z2[nn, ...] + n_c) * (n_c - _z2 * D3_z2[nn - 1, ...])
        den = (_z1 * D1_z1[nn, ...] + n_c) * (n_c - _z1 * D3_z1[nn - 1, ...])
        den = torch.where(den.abs() < eps, den + eps, den)
        Q_curr = Q_prev * x_ratio_sq * (num / den)
        Q_list.append(Q_curr)
        Q_prev = Q_curr

    return torch.stack(Q_list, dim=0)


def pena_psi_zeta_n(
    n: Union[torch.Tensor, int],
    z: torch.Tensor,
    D1: torch.Tensor,
    D3: torch.Tensor,
    eps: float = 1e-30,
    precision: str = "double",
):
    """Build ``psi_n`` and ``zeta_n`` sequences from Peña derivatives.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).
    D1, D3 : torch.Tensor
        Log-derivative sequences.
    eps : float, default=1e-30
        Small denominator safeguard.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.

    Returns
    -------
    tuple of torch.Tensor
        ``(psi_n, zeta_n)`` for orders ``0..n``.
    """
    n_max = _pena_nmax(n)
    _z = torch.atleast_1d(torch.as_tensor(z))

    if precision.lower() == "single":
        dtype_c = torch.complex64
    else:
        dtype_c = torch.complex128
    _z = _z.to(dtype=dtype_c)

    psi_prev = torch.sin(_z)
    zeta_prev = torch.sin(_z) - 1j * torch.cos(_z)
    psi_list = [psi_prev]
    zeta_list = [zeta_prev]

    z_safe = torch.where(_z.abs() < eps, _z + eps, _z)
    for nn in range(1, n_max + 1):
        psi_curr = psi_prev * ((nn / z_safe) - D1[nn - 1, ...])
        zeta_curr = zeta_prev * ((nn / z_safe) - D3[nn - 1, ...])
        psi_list.append(psi_curr)
        zeta_list.append(zeta_curr)
        psi_prev = psi_curr
        zeta_prev = zeta_curr

    return torch.stack(psi_list, dim=0), torch.stack(zeta_list, dim=0)


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
    """Torch-native ``j_n`` via downward recurrence.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).
    n_add : {"auto"} or int, default="auto"
        Extra starting depth for downward sweep.
    n_add_min : int, default=10
        Minimum automatic extra depth.
    n_add_max : int, default=35
        Maximum automatic extra depth.
    eps : float, default=1e-10
        Small-argument safeguard.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.

    Returns
    -------
    torch.Tensor
        ``j_n(z)`` for orders ``0..n``.
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
    """Torch-native ``j_n`` via continued-fraction ratios.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).
    n_add : {"auto"} or int, default="auto"
        Extra continued-fraction depth.
    max_n_add : int, default=50
        Upper bound for automatic extra depth.
    small_z : float, default=1e-8
        Threshold using small-argument series.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.

    Returns
    -------
    torch.Tensor
        ``j_n(z)`` for orders ``0..n``.
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
    """Torch-native ``y_n`` via upward recurrence.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).
    eps : float, default=1e-10
        Small-argument safeguard.

    Returns
    -------
    torch.Tensor
        ``y_n(z)`` for orders ``0..n``.
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
    """Torch-native spherical Hankel function ``h_n^(1)``.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    torch.Tensor
        ``h_n^(1)(z)`` for orders ``0..n``.
    """
    return sph_jn_torch(n, z, **kwargs) + 1j * sph_yn_torch(n, z, **kwargs)


# Ricatti-Bessel
def psi_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Torch-native Riccati-Bessel ``psi_n`` and derivative.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    tuple of torch.Tensor
        ``(psi_n, dpsi_n_dz)`` for orders ``0..n``.
    """
    jn = sph_jn_torch(n, z, **kwargs)
    jn_der = f_der(n, z, jn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    psi = _z * jn
    psi_der = jn + _z * jn_der

    return psi, psi_der


def chi_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Torch-native Riccati-Bessel ``chi_n`` and derivative.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    tuple of torch.Tensor
        ``(chi_n, dchi_n_dz)`` for orders ``0..n``.
    """
    yn = sph_yn_torch(n, z, **kwargs)
    yn_der = f_der(n, z, yn)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    chi = -_z * yn
    chi_der = -yn - _z * yn_der

    return chi, chi_der


def xi_torch(n: torch.Tensor, z: torch.Tensor, **kwargs):
    """Torch-native Riccati-Bessel ``xi_n`` and derivative.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    z : torch.Tensor
        Complex argument(s).

    Returns
    -------
    tuple of torch.Tensor
        ``(xi_n, dxi_n_dz)`` for orders ``0..n``.
    """
    h1n = sph_h1n_torch(n, z, **kwargs)
    h1n_der = f_der(n, z, h1n)

    _n, _z = _expand_n_z(n, z)  # expand _z for broadcasting
    xin = _z * h1n
    xin_der = h1n + _z * h1n_der

    return xin, xin_der


# angular functions
def pi_tau(n: int, mu: torch.Tensor, **kwargs):
    """Compute angular functions ``pi_n`` and ``tau_n`` up to order ``n``.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum order.
    mu : torch.Tensor
        Cosine of polar angle.

    Returns
    -------
    tuple of torch.Tensor
        ``(pi, tau)`` with order axis first (including order 0).
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
    """Compute vector spherical harmonics with standard radial kernels.

    Parameters
    ----------
    n_max : int
        Maximum multipole order.
    k0 : torch.Tensor
        Vacuum wavevector.
    n_medium : torch.Tensor
        Medium refractive index.
    r, theta, phi : torch.Tensor
        Spherical coordinates.
    kind : int
        Radial kernel selector: ``1`` for ``j_n``, ``2`` for ``y_n``,
        ``3`` for ``h_n^(1)``.
    epsilon : float, default=1e-8
        Small origin safeguard.

    Returns
    -------
    tuple of torch.Tensor
        ``(M_o1n, M_e1n, N_o1n, N_e1n)``.
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


def D1n_torch(
    n: torch.Tensor,
    z: torch.Tensor,
    n_add="auto",
    n_add_min=10,
    n_add_max=35,
    eps=1e-10,
    precision="double",
    **kwargs,
):
    """Vectorized D(1)n logrithmic derivative via downward recurrence

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

    D1ns = []

    D1_n = torch.ones_like(_z, dtype=dtype_c) * 0.0
    D1_nm1 = torch.zeros_like(_z, dtype=dtype_c)

    for _n in range(n_max + n_add, 0, -1):
        D1_nm1 = _n/_z - 1/(D1_n + _n/_z)
        D1_n = D1_nm1
        if _n <= n_max + 1:
            D1ns.append(D1_n[-1, ...])


	# inverse order and convert to tensor
    D1ns = torch.stack(D1ns[::-1], dim=0)  # first dim: order n
    return D1ns



def D3n_torch(
    n: torch.Tensor,
    z: torch.Tensor,
    D1ns: torch.Tensor,
    eps=1e-10,
    precision="double",
    **kwargs,
):
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

    # instead splitting z into real and imag, could we just use exp
    # below is from top right page 5 of Mackowski et al (https://doi.org/10.1364/AO.29.001551)
    psixi0 = -1j * torch.exp(1j*z)*torch.sin(z)
    # psixi0 = 0.5*(1 - (torch.cos(2*_z[-1, ...].real) + 1j*torch.sin(2*_z[-1, ...].real)*torch.exp(-2*_z[-1, ...].imag)))
    D3ns = []

    D3ns.append(1j*torch.ones_like(_z[-1, ...]))

    psixis = []

    psixis.append(psixi0)

    if n_max > 0:
        for n_iter in range(1, n_max + 1):
            psixis.append(
                psixis[n_iter - 1] * ((n_iter/_z[-1, ...]) - D1ns[n_iter - 1]) * ((n_iter/_z[-1, ...]) - D3ns[n_iter - 1])
            )

            D3ns.append(
                D1ns[n_iter] + 1j/psixis[n_iter]
            )

    D3ns = torch.stack(D3ns, dim=0)  # first dim: order n
    psixis = torch.stack(psixis, dim=0)  # first dim: order n

    return D3ns, psixis


def Ql_torch(
    n: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    x: torch.Tensor,
    eps=1e-10,
    **kwargs
):
    # TODO make automatic using
    # Wiscombe, Warren J. "Improved Mie scattering algorithms."
    # Applied optics 19.9 (1980): 1505-1509.
    n_max = 10

    l_max = 10

    Qls = []

    for l_iter in range(1, l_max + 1):

        # get z1 and z2 for coresponding l TODO fix vectorisation
        z1 = m[:,l_iter-1,...] * x[:,l_iter-2,...] # or m[-1,l_iter-1,...] ???
        z2 = m[:,l_iter-1,...] * x[:,l_iter-1,...]

        # calculate D1 and D3 for z1 and z2

        D1n_z1 = D1n_torch(n, z1) # downwards recurrence
        D1n_z2 = D1n_torch(n, z2) # downwards recurrence

        D3n_z1 = D3n_torch(n, z1, D1n_z1) # upwards recurrence
        D3n_z2 = D3n_torch(n, z2, D1n_z2) # upwards recurrence

        # use z1 and z2 for zero order
        Qls.append(
            ((torch.exp(-2j*z1[-1, ...].real)-torch.exp(-2j*z1[-1, ...].imag))/(torch.exp(-2j*z2[-1, ...].real)-torch.exp(-2j*z2[-1, ...].real))) * torch.exp(-2*(z2[-1, ...].imag-z1[-1, ...].imag))
        )

        # use z1 and z2 for higher orders, upwards recurrence
        if n_max > 0:
            for n_iter in range(1, n_max + 1):
                Qls.append(
                    Qls[n_iter-1] * torch.square(x[l_iter-2]/x[l_iter-1]) \
                        * ((z2*D1n_z2[n_iter-1] + n_iter)/(z1*D1n_z1[n_iter-1] + n_iter)) \
                        * ((n_iter - z2*D3n_z2[n_iter-2])/(n_iter - z1*D3n_z1[n_iter-2]))
                )

        # Do somthing for each l, append to 2d list? I dont know how to avoid in-place modif. error


    Qls = torch.stack(Qls, dim=0) # TODO fix vectorisation

    return Qls


def psi_torch_logdir(
    n: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    x: torch.Tensor,
    eps=1e-10,
    **kwargs
):
    # TODO make automatic using
    # Wiscombe, Warren J. "Improved Mie scattering algorithms."
    # Applied optics 19.9 (1980): 1505-1509.
    n_max = 10

    psis = []

    xL = x[-1, ...] # get last size parameter i.e. l=L

    D1n_xL = D1n_torch(n, xL) # downwards recurrence

    # zero order
    psis.append(
        torch.sin(xL)
    )

    if n_max > 0:
        for n_iter in range(1, n_max + 1):
            psis.append(
                psis[n_iter-2] * (n_iter/xL - D1n_xL[n_iter-2])
            )

    psis = torch.stack(psis, dim=0) # first dim: order n (just n as only one l: L)


def xi_torch_logdir(
    n: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    x: torch.Tensor,
    eps=1e-10,
    **kwargs
):
    # TODO make automatic using
    # Wiscombe, Warren J. "Improved Mie scattering algorithms."
    # Applied optics 19.9 (1980): 1505-1509.
    n_max = 10
    l_max = 10

    xis = []

    xL = x[-1, ...] # get last size parameter i.e. l=L NOTE !This could be done outside the function!

    D1n_xL = D1n_torch(n, xL) # downwards recurrence
    D3n_xL = D3n_torch(n, xL, D1n_xL) # upwards recurrence

    #zero order
    xis.append(
        torch.sin(xL) - 1j*torch.cos(xL)
    )

    if n_max > 0:
        for n_iter in range(1, n_max + 1):
            xis.append(
                xis[n_iter-2]*(n_iter/xL - D3n_xL[n_iter-2])
            )

    xis = torch.stack(xis, dim=0) # first dim: order n (just n as only one l: L)



def vsh_pena(
    n_max,
    k0,
    n_medium,
    r,
    theta,
    phi,
    kind,
    precision="double",
    epsilon=1e-8,
):
    """Compute VSH terms using Peña log-derivative recurrences.

    Parameters
    ----------
    n_max : int
        Maximum multipole order.
    k0 : torch.Tensor
        Vacuum wavevector.
    n_medium : torch.Tensor
        Medium refractive index.
    r, theta, phi : torch.Tensor
        Spherical coordinates.
    kind : int
        Radial kernel selector: ``1`` for ``psi_n`` and ``3`` for ``zeta_n``.
    precision : {"single", "double"}, default="double"
        Complex dtype selection.
    epsilon : float, default=1e-8
        Small origin safeguard.

    Returns
    -------
    tuple of torch.Tensor
        ``(M_o1n, M_e1n, N_o1n, N_e1n)``.
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cos_p = torch.cos(phi)
    sin_p = torch.sin(phi)
    rho = k0 * n_medium * r + epsilon

    # angular functions
    pi_n, tau_n = pi_tau(n_max, cos_t[0])
    pi_n = pi_n[1:]
    tau_n = tau_n[1:]

    # radial Riccati-Bessel functions and log-derivatives
    D1 = pena_D1_n(n_max, rho[0], precision=precision)
    D3 = pena_D3_n(n_max, rho[0], D1=D1, precision=precision)
    psi_n, zeta_n = pena_psi_zeta_n(n_max, rho[0], D1=D1, D3=D3, precision=precision)

    if kind == 1:
        rho_zn = psi_n[1:]
        D_kind = D1[1:]
    elif kind == 3:
        rho_zn = zeta_n[1:]
        D_kind = D3[1:]
    else:
        raise ValueError("`kind` parameter must be either 1 or 3 for `vsh_pena`.")

    zn = rho_zn / rho
    rho_zn_der_over_rho = D_kind * zn

    # spherical coordinate basis vectors
    e_r = torch.as_tensor([1, 0, 0], device=k0.device).view((r.ndim - 1) * (1,) + (-1,))
    e_tet = torch.as_tensor([0, 1, 0], device=k0.device).view(
        (r.ndim - 1) * (1,) + (-1,)
    )
    e_phi = torch.as_tensor([0, 0, 1], device=k0.device).view(
        (r.ndim - 1) * (1,) + (-1,)
    )

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
