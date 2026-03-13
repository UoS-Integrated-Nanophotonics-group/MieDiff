# -*- coding: utf-8 -*-
"""
pure torch implementations of spherical Bessel function recurrences

"""
# %%
import warnings

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


# --- torch-native spherical Bessel functions via recurrences
def sph_jn_torch_rec(
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


def sph_yn_torch_rec(n: torch.Tensor, z: torch.Tensor, eps=1e-10, **kwargs):
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


# --------------------------------
# logarithmic derivatives
# --------------------------------
def log_deriv_psn_torch(n, z, eps=1e-10, **kwargs):
    _n, _z = _expand_n_z(n, z)
    n_max = _n.max().item()
    assert n_max >= 0

    sin_z = torch.sin(_z[-1, ...])
    cos_z = torch.cos(_z[-1, ...])
    sin_z = torch.where(sin_z.abs() < eps, eps * torch.ones_like(sin_z), sin_z)
    L = cos_z / sin_z
    results = [L]

    for i in range(1, n_max + 1):
        term = i / _z[-1, ...] - results[-1]
        term = torch.where(term.abs() < eps, eps * torch.ones_like(term), term)
        L = 1 / term - i / _z[-1, ...]
        results.append(L)

    return torch.stack(results, dim=0)


def log_deriv_xin_torch(n, z, eps=1e-10, **kwargs):
    _n, _z = _expand_n_z(n, z)
    n_max = _n.max().item()
    assert n_max >= 0

    M = torch.ones_like(_z[-1, ...]) * complex(0, 1)
    results = [M]

    for i in range(1, n_max + 1):
        term = i / _z[-1, ...] - results[-1]
        term = torch.where(term.abs() < eps, eps * torch.ones_like(term), term)
        M = 1 / term - i / _z[-1, ...]
        results.append(M)

    return torch.stack(results, dim=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import spherical_jn, spherical_yn

    # Test values
    z_test = torch.tensor([1.5, 2j, 3.0-5j], dtype=torch.complex128)
    n_max = 2  # Test up to order 2

    # Compute using our PyTorch functions
    log_deriv_psn = log_deriv_psn_torch(n_max, z_test)
    log_deriv_xin = log_deriv_xin_torch(n_max, z_test)

    # Compute using SciPy for comparison
    def compute_log_deriv_psn_scipy(n_max, z):
        results = []
        for n in range(n_max + 1):
            j_n = spherical_jn(n, z)
            j_n_plus_1 = spherical_jn(n + 1, z)
            psi_n = z * j_n
            psi_deriv = (n + 1) * j_n - z * j_n_plus_1
            log_deriv = psi_deriv / psi_n
            results.append(log_deriv)
        return np.array(results)

    def compute_log_deriv_xin_scipy(n_max, z):
        results = []
        for n in range(n_max + 1):
            j_n = spherical_jn(n, z)
            y_n = spherical_yn(n, z)
            h_n = j_n + 1j * y_n
            xi_n = z * h_n
            # Compute h_{n+1}(z)
            j_n_plus_1 = spherical_jn(n + 1, z)
            y_n_plus_1 = spherical_yn(n + 1, z)
            h_n_plus_1 = j_n_plus_1 + 1j * y_n_plus_1
            xi_deriv = (n + 1) * h_n - z * h_n_plus_1
            log_deriv = xi_deriv / xi_n
            results.append(log_deriv)
        return np.array(results)

    # Convert torch tensors to numpy arrays for SciPy
    z_np = z_test.numpy()

    # Compute logarithmic derivatives using SciPy
    log_deriv_psn_scipy = compute_log_deriv_psn_scipy(n_max, z_np)
    log_deriv_xin_scipy = compute_log_deriv_xin_scipy(n_max, z_np)

    # Compare results
    print("Logarithmic derivatives for psi_n(z):")
    # print("PyTorch:")
    # print(log_deriv_psn)
    # print("SciPy:")
    # print(log_deriv_psn_scipy)
    print("Difference:")
    print(log_deriv_psn - log_deriv_psn_scipy)

    print("\nLogarithmic derivatives for xi_n(z):")
    # print("PyTorch:")
    # print(log_deriv_xin)
    # print("SciPy:")
    # print(log_deriv_xin_scipy)
    print("Difference:")
    print(log_deriv_xin - log_deriv_xin_scipy)

    torch.testing.assert_close(log_deriv_xin.numpy(), log_deriv_xin_scipy)
    torch.testing.assert_close(log_deriv_psn.numpy(), log_deriv_psn_scipy)
