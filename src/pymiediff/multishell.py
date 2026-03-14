# -*- coding: utf-8 -*-
"""Vectorized Mie solvers for spherical particles.

This module implements low-level numerical routines for:

- Mie coefficients (`a_n`, `b_n`, optional internal coefficients),
- integrated observables (cross sections),
- far-field angular scattering amplitudes,
- near-field electric and magnetic components.

The API is multilayer-first while keeping compatibility with legacy
homogeneous/core-shell arguments.
"""
import warnings

import torch
import numpy as np

from pymiediff import special
from pymiediff import helper


# - Mie coefficients, private API
def _miecoef(
    x,
    y,
    n,
    m1,
    m2,
    return_internal=False,
    backend="torch",
    precision="double",
    which_jn="recurrence",
):
    """Compute Mie coefficients for homogeneous or core-shell particles.

    Parameters
    ----------
    x, y : torch.Tensor
        Inner and outer size parameters in the host medium.
    n : int or torch.Tensor
        Truncation order.
    m1, m2 : torch.Tensor
        Relative refractive indices for core and shell.
    return_internal : bool, default=False
        If ``True``, also compute internal coefficients.
    backend : {"torch", "scipy"}, default="torch"
        Special-function backend.
    precision : {"single", "double"}, default="double"
        Precision mode for torch-based recurrences.
    which_jn : {"recurrence", "ratios"}, default="recurrence"
        Torch-only algorithm for spherical ``j_n``.

    Returns
    -------
    dict
        Dictionary containing at least ``a_n`` and ``b_n``.
    """
    # backend selection
    if backend.lower() in ["torch"] and which_jn.lower() != "recurrence":
        sph_jn_func = special.sph_jn_torch
        sph_yn_func = special.sph_yn_torch
    elif backend.lower() in ["torch"] and which_jn.lower() == "recurrence":
        sph_jn_func = special.sph_jn_torch_via_rec
        sph_yn_func = special.sph_yn_torch
    elif backend.lower() in ["scipy"]:
        sph_jn_func = special.sph_jn
        sph_yn_func = special.sph_yn
    else:
        raise ValueError("Unknown backend configuration.")

    # permeabilities are set to 1 for now
    mu1 = mu2 = mu = 1.0

    # evaluate bessel terms
    j_y = sph_jn_func(n, y, precision=precision)
    y_y = sph_yn_func(n, y, precision=precision)
    j_m1x = sph_jn_func(n, m1 * x, precision=precision)
    j_m2x = sph_jn_func(n, m2 * x, precision=precision)
    j_m2y = sph_jn_func(n, m2 * y, precision=precision)

    y_m2x = sph_yn_func(n, m2 * x, precision=precision)
    y_m2y = sph_yn_func(n, m2 * y, precision=precision)

    h1_y = j_y + 1j * y_y

    # bessel derivatives
    dj_y = special.f_der(n, y, j_y, precision=precision)
    dj_m1x = special.f_der(n, m1 * x, j_m1x, precision=precision)
    dj_m2x = special.f_der(n, m2 * x, j_m2x, precision=precision)
    dj_m2y = special.f_der(n, m2 * y, j_m2y, precision=precision)
    dy_y = special.f_der(n, y, y_y, precision=precision)
    dy_m2x = special.f_der(n, m2 * x, y_m2x, precision=precision)
    dy_m2y = special.f_der(n, m2 * y, y_m2y, precision=precision)

    dh1_y = dj_y + 1j * dy_y

    # eval. ricatti-bessel terms (psi, chi, xi)
    psi_y = y * j_y
    psi_m1x = (m1 * x) * j_m1x
    psi_m2x = (m2 * x) * j_m2x
    psi_m2y = (m2 * y) * j_m2y

    chi_m2x = -(m2 * x) * y_m2x
    chi_m2y = -(m2 * y) * y_m2y

    xi_y = y * h1_y

    dpsi_y = j_y + y * dj_y
    dpsi_m1x = j_m1x + (m1 * x) * dj_m1x
    dpsi_m2x = j_m2x + (m2 * x) * dj_m2x
    dpsi_m2y = j_m2y + (m2 * y) * dj_m2y

    dchi_m2x = -y_m2x - (m2 * x) * dy_m2x
    dchi_m2y = -y_m2y - (m2 * y) * dy_m2y

    dxi_y = h1_y + y * dh1_y

    # # Mie coeffs.
    # An = (m2 * psi_m2x * dpsi_m1x - m1 * dpsi_m2x * psi_m1x) / (
    #     m2 * chi_m2x * dpsi_m1x - m1 * dchi_m2x * psi_m1x
    # )
    # Bn = (m2 * psi_m1x * dpsi_m2x - m1 * psi_m2x * dpsi_m1x) / (
    #     m2 * dchi_m2x * psi_m1x - m1 * dpsi_m1x * chi_m2x
    # )

    # an = (
    #     psi_y * (dpsi_m2y - An * dchi_m2y) - m2 * dpsi_y * (psi_m2y - An * chi_m2y)
    # ) / (xi_y * (dpsi_m2y - An * dchi_m2y) - m2 * dxi_y * (psi_m2y - An * chi_m2y))
    # bn = (
    #     m2 * psi_y * (dpsi_m2y - Bn * dchi_m2y) - dpsi_y * (psi_m2y - Bn * chi_m2y)
    # ) / (m2 * xi_y * (dpsi_m2y - Bn * dchi_m2y) - dxi_y * (psi_m2y - Bn * chi_m2y))

    # common expressions - scattering coefficients
    # (Via sympy, solving Bohren Huffmann Eq 8.1)
    x0 = dpsi_m2y * psi_y
    x1 = chi_m2x * dpsi_m1x
    x2 = m2 * mu2
    x3 = mu1 * x2
    x4 = x1 * x3
    x5 = m1 * mu2**2
    x6 = dpsi_m2x * psi_m1x
    x7 = dchi_m2y * psi_y
    x8 = x6 * x7
    x9 = dpsi_m1x * psi_m2x
    x10 = x3 * x9
    x11 = dchi_m2x * psi_m1x
    x12 = psi_m2y * x11
    x13 = m1 * x2
    x14 = mu * x13
    x15 = x12 * x14
    x16 = x11 * x5
    x17 = m2**2 * mu1
    x18 = x17 * x9
    x19 = chi_m2y * dpsi_y
    x20 = mu * x19
    x21 = x13 * x6
    x22 = chi_m2x * psi_m2y
    x23 = mu * x17
    x24 = dpsi_m1x * x22 * x23
    x25 = dpsi_m2y * xi_y
    x26 = dchi_m2y * xi_y
    x27 = x26 * x6
    x28 = chi_m2y * dxi_y
    x29 = mu * x28
    x30 = 1 / (
        dxi_y * x15
        - dxi_y * x24
        - x10 * x26
        - x16 * x25
        + x18 * x29
        - x21 * x29
        + x25 * x4
        + x27 * x5
    )
    x31 = x12 * x3
    x32 = dpsi_m1x * x5
    x33 = psi_m2x * x32
    x34 = x3 * x6
    x35 = x1 * x14
    x36 = x22 * x32
    x37 = x14 * x9
    x38 = x11 * x23
    x39 = 1 / (
        dxi_y * x31
        - dxi_y * x36
        + x23 * x27
        + x25 * x35
        - x25 * x38
        - x26 * x37
        + x28 * x33
        - x28 * x34
    )

    # scattering coefficients (external)
    an = x30 * (
        dpsi_y * x15
        - dpsi_y * x24
        - x0 * x16
        + x0 * x4
        - x10 * x7
        + x18 * x20
        - x20 * x21
        + x5 * x8
    )
    bn = x39 * (
        dpsi_y * x31
        - dpsi_y * x36
        + x0 * x35
        - x0 * x38
        + x19 * x33
        - x19 * x34
        + x23 * x8
        - x37 * x7
    )

    if return_internal:
        # common expressions - internal coefficients
        x40 = x2 * x39
        x41 = chi_m2x * dpsi_m2x
        x42 = dpsi_y * xi_y
        x43 = dxi_y * psi_y
        x44 = dchi_m2x * psi_m2x
        x45 = m1 * mu1 * (x41 * x42 - x41 * x43 - x42 * x44 + x43 * x44)
        x46 = x2 * x30

        x47 = m1 * mu2
        x48 = x1 * x42
        x49 = m2 * mu1
        x50 = x11 * x43
        x51 = x1 * x43
        x52 = x11 * x42
        x53 = x42 * x47
        x54 = x49 * x6
        x55 = x43 * x47
        x56 = x49 * x9

        # internal coefficients (core)
        cn = x40 * x45
        dn = x45 * x46

        # internal coefficients (shell)
        fn = x40 * (x47 * x48 - x47 * x51 + x49 * x50 - x49 * x52)
        gn = x46 * (x47 * x50 - x47 * x52 + x48 * x49 - x49 * x51)
        vn = x40 * (-x42 * x54 + x43 * x54 + x53 * x9 - x55 * x9)
        wn = x46 * (x42 * x56 - x43 * x56 - x53 * x6 + x55 * x6)

    # recurrences return n+1 orders: remove zeroth order
    if return_internal:
        result_dict = dict(
            a_n=an[1:, ...],
            b_n=bn[1:, ...],
            c_n=cn[1:, ...],
            d_n=dn[1:, ...],
            f_n=fn[1:, ...],
            g_n=gn[1:, ...],
            v_n=vn[1:, ...],
            w_n=wn[1:, ...],
        )
    else:
        result_dict = dict(
            a_n=an[1:, ...],
            b_n=bn[1:, ...],
        )

    return result_dict


def _miecoef_pena(
    k,
    r_layers,
    eps_layers,
    n_env,
    n,
    return_internal=False,
    precision="double",
):
    """Compute external multilayer Mie coefficients with Peña/Yang recurrences.

    Parameters
    ----------
    k : torch.Tensor
        Host-medium wavevector spectrum.
    r_layers : torch.Tensor
        Layer radii, shape ``(N_part, L)``.
    eps_layers : torch.Tensor
        Layer permittivities, shape ``(N_part, L, N_k0)``.
    n_env : torch.Tensor
        Host refractive index.
    n : int or torch.Tensor
        Truncation order.
    return_internal : bool, default=False
        Not supported in this backend.
    precision : {"single", "double"}, default="double"
        Computation precision.

    Returns
    -------
    dict
        Dictionary with ``a_n`` and ``b_n``.
    """
    if return_internal:
        raise NotImplementedError(
            "backend='pena' currently supports only external coefficients (a_n, b_n)."
        )

    n_max = int(torch.as_tensor(n).item())
    if n_max < 1:
        raise ValueError("`n` must be >= 1.")

    if precision.lower() == "single":
        dtype_c = torch.complex64
    else:
        dtype_c = torch.complex128

    eps = torch.tensor(1e-30, device=k.device, dtype=dtype_c)

    k = torch.as_tensor(k).to(dtype=dtype_c)
    r_layers = torch.as_tensor(r_layers, device=k.device)
    eps_layers = torch.as_tensor(eps_layers, device=k.device).to(dtype=dtype_c)
    n_env = torch.as_tensor(n_env, device=k.device).to(dtype=dtype_c)

    # x_l = k * r_l in host medium, m_l relative to host
    x_layers = k.unsqueeze(1) * r_layers.unsqueeze(-1).to(dtype=dtype_c)
    m_layers = torch.sqrt(eps_layers) / n_env.unsqueeze(1)
    n_part, n_layers, n_k0 = x_layers.shape

    H_a = None
    H_b = None

    # layer recursion for H^a and H^b (Peña/Pal 2009, Yang recursion)
    for l_idx in range(n_layers):
        ml = m_layers[:, l_idx, :]
        xl = x_layers[:, l_idx, :]
        z_curr = ml * xl

        D1_curr = special.pena_D1_n(n_max, z_curr, precision=precision)
        D3_curr = special.pena_D3_n(
            n_max, z_curr, D1=D1_curr, precision=precision
        )

        if l_idx == 0:
            H_a = D1_curr
            H_b = D1_curr.clone()
            continue

        mlm1 = m_layers[:, l_idx - 1, :]
        z_prev = ml * x_layers[:, l_idx - 1, :]
        D1_prev = special.pena_D1_n(n_max, z_prev, precision=precision)
        D3_prev = special.pena_D3_n(
            n_max, z_prev, D1=D1_prev, precision=precision
        )
        Q = special.pena_Q_n(
            n_max,
            z1=z_prev,
            z2=z_curr,
            x1=x_layers[:, l_idx - 1, :],
            x2=x_layers[:, l_idx, :],
            D1_z1=D1_prev,
            D1_z2=D1_curr,
            D3_z1=D3_prev,
            D3_z2=D3_curr,
            precision=precision,
        )

        G1 = ml * H_a - mlm1 * D1_prev
        G2 = ml * H_a - mlm1 * D3_prev
        den_a = G2 - Q * G1
        den_a = torch.where(den_a.abs() < eps.abs(), den_a + eps, den_a)
        H_a = (G2 * D1_curr - Q * G1 * D3_curr) / den_a

        Gt1 = mlm1 * H_b - ml * D1_prev
        Gt2 = mlm1 * H_b - ml * D3_prev
        den_b = Gt2 - Q * Gt1
        den_b = torch.where(den_b.abs() < eps.abs(), den_b + eps, den_b)
        H_b = (Gt2 * D1_curr - Q * Gt1 * D3_curr) / den_b

    # final scattering coefficients with outer size parameter x_L
    xL = x_layers[:, -1, :]
    mL = m_layers[:, -1, :]
    D1_xL = special.pena_D1_n(n_max, xL, precision=precision)
    D3_xL = special.pena_D3_n(n_max, xL, D1=D1_xL, precision=precision)
    psi_xL, zeta_xL = special.pena_psi_zeta_n(
        n_max, xL, D1=D1_xL, D3=D3_xL, precision=precision
    )

    n_arr = torch.arange(n_max + 1, device=k.device, dtype=k.real.dtype).to(dtype_c)
    n_arr = n_arr.view(-1, 1, 1)
    xL_safe = torch.where(xL.abs() < eps.abs(), xL + eps, xL)

    term_a = H_a / mL.unsqueeze(0) + (n_arr / xL_safe.unsqueeze(0))
    den_a = term_a[1:, ...] * zeta_xL[1:, ...] - zeta_xL[:-1, ...]
    den_a = torch.where(den_a.abs() < eps.abs(), den_a + eps, den_a)
    a_n = (
        term_a[1:, ...] * psi_xL[1:, ...] - psi_xL[:-1, ...]
    ) / den_a

    term_b = mL.unsqueeze(0) * H_b + (n_arr / xL_safe.unsqueeze(0))
    den_b = term_b[1:, ...] * zeta_xL[1:, ...] - zeta_xL[:-1, ...]
    den_b = torch.where(den_b.abs() < eps.abs(), den_b + eps, den_b)
    b_n = (
        term_b[1:, ...] * psi_xL[1:, ...] - psi_xL[:-1, ...]
    ) / den_b

    return dict(
        a_n=a_n,
        b_n=b_n,
    )


# - internal helper
def _broadcast_mie_config(k0, r_c, r_s, eps_c, eps_s, eps_env):
    """Broadcast legacy core/shell inputs to vectorized tensor shapes.

    Parameters
    ----------
    k0, r_c, r_s, eps_c, eps_s, eps_env : tensor-like
        Legacy solver inputs.

    Returns
    -------
    tuple
        Broadcast tensors in ``(N_part, N_k0)`` convention
        (with ``k0``/``eps_env`` having leading singleton particle axis).
    """
    # convert everything to tensors
    k0 = torch.as_tensor(k0)
    k0 = k0.squeeze()  # remove possible empty dimensions
    k0 = torch.atleast_1d(k0)  # if single value, expand (=spectrum)
    assert len(k0.shape) == 1

    # add N particle dimension
    k0 = k0.unsqueeze(0)

    # input shape r_c,s: N particles
    # input shape eps_env: N wavelengths
    # input shape eps_c,s: (N particles, N wavelengths)
    r_c = torch.as_tensor(r_c)
    r_c = torch.atleast_1d(r_c)  # if single particle, expand
    assert len(r_c.shape) == 1
    r_c = r_c.unsqueeze(-1)  # add wavelength dimension
    r_c = r_c.broadcast_to((r_c.shape[0], k0.shape[1]))

    r_s = torch.as_tensor(r_s)
    r_s = torch.atleast_1d(r_s)  # if single particle, expand
    assert len(r_s.shape) == 1
    r_s = r_s.unsqueeze(-1)  # add wavelength dimension
    r_s = r_s.broadcast_to((r_s.shape[0], k0.shape[1]))
    assert r_c.shape == r_s.shape

    eps_c = torch.as_tensor(eps_c)
    eps_c = torch.atleast_1d(eps_c)
    if eps_c.dim() == 1 and len(eps_c) == len(r_c):
        eps_c = eps_c.broadcast_to((r_c.shape[0], k0.shape[1]))
    else:
        eps_c = eps_c.reshape((r_c.shape[0], k0.shape[1]))

    eps_s = torch.as_tensor(eps_s)
    eps_s = torch.atleast_1d(eps_s)
    if eps_s.dim() == 1 and len(eps_s) == len(r_s):
        eps_s = eps_s.broadcast_to((r_s.shape[0], k0.shape[1]))
    else:
        eps_s = eps_s.reshape((r_s.shape[0], k0.shape[1]))

    assert r_c.shape[0] == r_s.shape[0]
    assert eps_c.shape[0] == r_c.shape[0]
    assert eps_s.shape[0] == r_c.shape[0]
    assert eps_c.shape[1] == k0.shape[1]
    assert eps_s.shape[1] == k0.shape[1]

    # input shape should be as k0
    eps_env = torch.as_tensor(eps_env)
    eps_env = torch.atleast_1d(eps_env).unsqueeze(0)  # particle dim.

    return k0, r_c, r_s, eps_c, eps_s, eps_env


def _as_layer_inputs(r_c, r_s, eps_c, eps_s, r_layers=None, eps_layers=None):
    """Normalize legacy core/shell inputs to layer arrays.

    Parameters
    ----------
    r_c, r_s, eps_c, eps_s : tensor-like or None
        Legacy core/shell inputs.
    r_layers, eps_layers : tensor-like or None
        Multilayer inputs.

    Returns
    -------
    tuple
        ``(r_layers, eps_layers)`` in layer-major representation.
    """
    if (r_layers is None) != (eps_layers is None):
        raise ValueError("`r_layers` and `eps_layers` must be provided together.")

    if r_layers is not None and eps_layers is not None:
        return torch.as_tensor(r_layers), torch.as_tensor(eps_layers)

    if r_c is None or eps_c is None:
        raise ValueError(
            "Either provide (`r_layers`, `eps_layers`) or legacy (`r_c`, `eps_c`)."
        )

    # legacy API mapping: homogeneous/core-shell -> L=1/2
    if r_s is None:
        r_s = r_c
    if eps_s is None:
        eps_s = eps_c

    r_c_t = torch.atleast_1d(torch.as_tensor(r_c))
    r_s_t = torch.atleast_1d(torch.as_tensor(r_s))
    if r_c_t.shape != r_s_t.shape:
        raise ValueError("`r_c` and `r_s` must have compatible shapes.")

    eps_c_t = torch.as_tensor(eps_c)
    eps_s_t = torch.as_tensor(eps_s)

    # choose L=1 when core and shell are exactly equal, otherwise L=2
    if torch.equal(r_c_t, r_s_t):
        r_layers = r_c_t.unsqueeze(-1)
        if eps_c_t.ndim == 0:
            # single particle, constant epsilon: (N_part, L)
            eps_layers = eps_c_t.view(1, 1)
        elif eps_c_t.ndim == 1:
            if eps_c_t.shape == r_c_t.shape and r_c_t.numel() > 1:
                # per-particle constant epsilon: (N_part, L)
                eps_layers = eps_c_t.unsqueeze(1)
            else:
                # single particle, spectral epsilon: (L, N_k0)
                eps_layers = eps_c_t.unsqueeze(0)
        else:
            # particle-batched layout: (N_part, L, N_k0)
            eps_layers = eps_c_t.unsqueeze(1)
    else:
        r_layers = torch.stack((r_c_t, r_s_t), dim=-1)
        if eps_c_t.ndim == 0:
            # single particle, constant eps in both regions: (N_part, L)
            eps_layers = torch.stack((eps_c_t, eps_s_t), dim=0).unsqueeze(0)
        elif eps_c_t.ndim == 1:
            if eps_c_t.shape == r_c_t.shape and r_c_t.numel() > 1:
                # per-particle constant eps: (N_part, L)
                eps_layers = torch.stack((eps_c_t, eps_s_t), dim=1)
            else:
                # single particle spectral eps: (L, N_k0)
                eps_layers = torch.stack((eps_c_t, eps_s_t), dim=0)
        else:
            # particle-batched spectral layout: (N_part, L, N_k0)
            eps_layers = torch.stack((eps_c_t, eps_s_t), dim=1)

    return r_layers, eps_layers


def _broadcast_mie_layers(k0, r_layers, eps_layers, eps_env):
    """Broadcast multilayer configuration for vectorized Mie calculations.

    Parameters
    ----------
    k0 : tensor-like
        Vacuum wavevector spectrum.
    r_layers : tensor-like
        Layer radii with shape ``(L,)`` or ``(N_part, L)``.
    eps_layers : tensor-like
        Layer permittivities in one of accepted shapes.
    eps_env : tensor-like
        Environment permittivity.

    Returns
    -------
    tuple
        ``(k0, r_layers, eps_layers, eps_env, n_layers_rel, n_env)``.
    """
    k0 = torch.as_tensor(k0).squeeze()
    k0 = torch.atleast_1d(k0)
    if k0.ndim != 1:
        raise ValueError("`k0` must be a 1D tensor-like object.")
    n_k0 = k0.shape[0]
    k0 = k0.unsqueeze(0)  # particle dim

    r_layers = torch.as_tensor(r_layers)
    r_layers = torch.atleast_1d(r_layers)
    if r_layers.ndim == 1:
        r_layers = r_layers.unsqueeze(0)
    if r_layers.ndim != 2:
        raise ValueError("`r_layers` must have shape (N_part, L) or (L,).")
    n_part, n_layers = r_layers.shape

    # enforce increasing layer radii
    if n_layers > 1 and not torch.all(r_layers[:, 1:] > r_layers[:, :-1]):
        raise ValueError("`r_layers` must be strictly increasing along layer axis.")

    eps_layers = torch.as_tensor(eps_layers)
    eps_layers = torch.atleast_1d(eps_layers)

    # accepted eps_layers layouts:
    #   (L,) / (N_part, L) / (L, N_k0) / (N_part, L, N_k0)
    if eps_layers.ndim == 1:
        if eps_layers.shape[0] != n_layers:
            raise ValueError("`eps_layers` with 1D shape must match number of layers.")
        eps_layers = eps_layers.view(1, n_layers, 1).broadcast_to(n_part, n_layers, n_k0)
    elif eps_layers.ndim == 2:
        if eps_layers.shape == (n_part, n_layers):
            eps_layers = eps_layers.unsqueeze(-1).broadcast_to(n_part, n_layers, n_k0)
        elif eps_layers.shape == (n_layers, n_k0):
            eps_layers = eps_layers.unsqueeze(0).broadcast_to(n_part, n_layers, n_k0)
        else:
            raise ValueError(
                "`eps_layers` 2D shape must be (N_part, L) or (L, N_k0)."
            )
    elif eps_layers.ndim == 3:
        if eps_layers.shape[0] == 1 and n_part > 1:
            eps_layers = eps_layers.broadcast_to(n_part, eps_layers.shape[1], eps_layers.shape[2])
        if eps_layers.shape[:2] != (n_part, n_layers):
            raise ValueError("`eps_layers` first two dimensions must be (N_part, L).")
        if eps_layers.shape[2] == 1 and n_k0 > 1:
            eps_layers = eps_layers.broadcast_to(n_part, n_layers, n_k0)
        elif eps_layers.shape[2] != n_k0:
            raise ValueError("`eps_layers` spectral dimension must match len(k0).")
    else:
        raise ValueError("`eps_layers` must be 1D, 2D, or 3D.")

    eps_env = torch.as_tensor(eps_env)
    eps_env = torch.atleast_1d(eps_env)
    if eps_env.ndim != 1:
        raise ValueError("`eps_env` must be scalar or 1D.")
    if eps_env.shape[0] == 1 and n_k0 > 1:
        eps_env = eps_env.broadcast_to(n_k0)
    elif eps_env.shape[0] != n_k0:
        raise ValueError("`eps_env` length must match len(k0).")
    eps_env = eps_env.unsqueeze(0)  # particle dim

    # layer-specific refractive index
    n_layers_rel = eps_layers**0.5
    n_env = eps_env**0.5

    return k0, r_layers, eps_layers, eps_env, n_layers_rel, n_env




# - Mie coefficients - public API
def mie_coefficients(
    k0,
    r_layers=None,
    eps_layers=None,
    r_c=None,
    eps_c=None,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    return_internal=False,
    backend="pena",
    precision="double",
    which_jn="recurrence",
    n_max=None,
):
    """Compute Mie coefficients for spherical particles.

    Parameters
    ----------
    k0 : tensor-like
        Vacuum wavevector(s), rad/nm.
    r_layers, eps_layers : tensor-like, optional
        Preferred multilayer inputs.
    r_c, eps_c, r_s, eps_s : tensor-like, optional
        Legacy core/shell inputs.
    eps_env : tensor-like, default=1.0
        Environment permittivity.
    return_internal : bool, default=False
        Return internal coefficients where supported.
    backend : {"pena", "torch", "scipy"}, default="pena"
        Numerical backend.
    precision : {"single", "double"}, default="double"
        Precision for torch-based recurrences.
    which_jn : {"recurrence", "ratios"}, default="recurrence"
        Torch-only spherical ``j_n`` implementation.
    n_max : int, optional
        Truncation order. Auto-estimated when omitted.

    Returns
    -------
    dict
        External coefficients ``a_n``/``b_n`` and solver metadata.
    """
    backend_l = backend.lower()
    if backend_l not in ("torch", "scipy", "pena"):
        raise ValueError("Unknown backend. Expected one of: 'torch', 'scipy', 'pena'.")

    # normalize all inputs to a common multilayer representation first
    r_layers, eps_layers = _as_layer_inputs(
        r_c=r_c,
        r_s=r_s,
        eps_c=eps_c,
        eps_s=eps_s,
        r_layers=r_layers,
        eps_layers=eps_layers,
    )
    k0, r_layers, eps_layers, eps_env, n_layers_rel, n_env = _broadcast_mie_layers(
        k0=k0,
        r_layers=r_layers,
        eps_layers=eps_layers,
        eps_env=eps_env,
    )
    n_layers_rel = eps_layers**0.5

    # compatibility aliases for legacy outputs and non-pena backends
    r_c = r_layers[:, 0].unsqueeze(-1).broadcast_to(r_layers.shape[0], k0.shape[1])
    if r_layers.shape[1] > 1:
        r_s = r_layers[:, -1].unsqueeze(-1).broadcast_to(
            r_layers.shape[0], k0.shape[1]
        )
    else:
        r_s = r_c
    eps_c = eps_layers[:, 0, :]
    if r_layers.shape[1] > 1:
        eps_s = eps_layers[:, -1, :]
    else:
        eps_s = eps_c

    n_c = eps_c**0.5
    n_s = eps_s**0.5

    # - Mie truncation order
    if n_max is None:
        # automatically determine truncation
        if backend_l == "pena":
            n_max = helper.get_truncution_criteroin_pena2009(
                k0=k0,
                r_layers=r_layers,
                eps_layers=eps_layers,
                eps_env=eps_env,
            )
        else:
            r_outer = r_layers[:, -1].unsqueeze(-1).broadcast_to(
                r_layers.shape[0], k0.shape[1]
            )
            ka = r_outer * k0 * torch.sqrt(eps_env)
            n_max = helper.get_truncution_criteroin_wiscombe(ka)
    n_max = torch.as_tensor(n_max, device=k0.device)
    n = torch.arange(1, n_max + 1, device=k0.device)

    # - eval Mie coefficients
    k = k0 * n_env
    if backend_l == "pena":
        mie_coef_result = _miecoef_pena(
            k=k,
            r_layers=r_layers,
            eps_layers=eps_layers,
            n_env=n_env,
            n=n_max,
            return_internal=return_internal,
            precision=precision,
        )
    else:
        if r_layers.shape[1] > 2:
            raise ValueError(
                "Backends 'torch' and 'scipy' support only homogeneous/core-shell inputs. "
                "Use backend='pena' for multilayer spheres."
            )
        x = k * r_c
        y = k * r_s
        m_c = n_c / n_env
        m_s = n_s / n_env

        # this will return order 1 to n_max (no zero order!)
        mie_coef_result = _miecoef(
            x=x,
            y=y,
            n=n_max,
            m1=m_c,
            m2=m_s,
            return_internal=return_internal,
            backend=backend,
            precision=precision,
            which_jn=which_jn,
        )

    return_dict = dict(
        k=k,
        k0=k0,
        n=n,
        n_max=n_max,
        r_c=r_c,
        r_s=r_s,
        eps_c=eps_c,
        eps_s=eps_s,
        eps_env=eps_env,
        n_c=n_c,
        n_s=n_s,
        n_env=n_env,
        r_layers=r_layers,
        eps_layers=eps_layers,
        m_layers=n_layers_rel / n_env.unsqueeze(1),
        L=torch.as_tensor(r_layers.shape[1], device=k0.device),
    )

    for k in mie_coef_result:
        return_dict[k] = mie_coef_result[k]

    return return_dict


# - Observables
def cross_sections(
    k0,
    r_layers=None,
    eps_layers=None,
    r_c=None,
    eps_c=None,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="pena",
    precision="double",
    which_jn="recurrence",
    n_max=None,
) -> dict:
    """Compute extinction, scattering, and absorption cross sections.

    Parameters
    ----------
    k0 : tensor-like
        Vacuum wavevector(s), rad/nm.
    r_layers, eps_layers, r_c, eps_c, r_s, eps_s, eps_env : tensor-like
        Geometry/material inputs (multilayer-first with legacy fallback).
    backend, precision, which_jn, n_max
        Forwarded to :func:`mie_coefficients`.

    Returns
    -------
    dict
        Total and multipole-resolved cross sections and efficiencies.
    """
    # - evaluate mie coefficients (vectorized)
    miecoeff = mie_coefficients(
        k0=k0,
        r_layers=r_layers,
        eps_layers=eps_layers,
        r_c=r_c,
        eps_c=eps_c,
        r_s=r_s,
        eps_s=eps_s,
        eps_env=eps_env,
        backend=backend,
        precision=precision,
        which_jn=which_jn,
        n_max=n_max,
    )
    n_max = miecoeff["n_max"]
    n = miecoeff["n"].unsqueeze(-1).unsqueeze(-1)  # add dim N_part, N_k0
    k = miecoeff["k"].unsqueeze(0)  # add dim order
    k0 = miecoeff["k0"].unsqueeze(0)  # add dim order
    r_s = miecoeff["r_s"].unsqueeze(0)  # add dim order
    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    # - geometric cross section
    cs_geo = torch.pi * r_s**2

    # - scattering efficiencies
    prefactor = 2 * torch.pi / (k**2)

    cs_ext_mp = prefactor * (2 * n + 1) * torch.stack((a_n.real, b_n.real))

    cs_sca_mp = prefactor * (2 * n + 1) * torch.stack((a_n.abs() ** 2, b_n.abs() ** 2))
    cs_abs_mp = cs_ext_mp - cs_sca_mp

    # full cross-sections:
    # sum multipole types (index 0) and multipole orders (index 1)
    cs_ext = torch.sum(cs_ext_mp, (0, 1)).real
    cs_abs = torch.sum(cs_abs_mp, (0, 1)).real
    cs_sca = torch.sum(cs_sca_mp, (0, 1)).real

    return dict(
        wavelength=2 * torch.pi / k0.squeeze(),
        k0=k0.squeeze(),
        n=n.squeeze(),
        n_max=n_max,
        cs_geo=cs_geo,
        # full cross sections
        q_ext=cs_ext / cs_geo[0, :],
        q_sca=cs_sca / cs_geo[0, :],
        q_abs=cs_abs / cs_geo[0, :],
        cs_ext=cs_ext,
        cs_sca=cs_sca,
        cs_abs=cs_abs,
        # separate multipoles
        q_ext_multipoles=cs_ext_mp.real / cs_geo,
        q_sca_multipoles=cs_sca_mp.real / cs_geo,
        q_abs_multipoles=cs_abs_mp.real / cs_geo,
        cs_ext_multipoles=cs_ext_mp.real,
        cs_sca_multipoles=cs_sca_mp.real,
        cs_abs_multipoles=cs_abs_mp.real,
    )


def angular_scattering(
    k0,
    theta,
    r_layers=None,
    eps_layers=None,
    r_c=None,
    eps_c=None,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    backend="pena",
    precision="double",
    which_jn="recurrence",
    n_max=None,
) -> dict:
    """Compute far-field scattering amplitudes and intensities.

    Parameters
    ----------
    k0 : tensor-like
        Vacuum wavevector(s), rad/nm.
    theta : tensor-like
        Scattering angles in radians.
    r_layers, eps_layers, r_c, eps_c, r_s, eps_s, eps_env : tensor-like
        Geometry/material inputs (multilayer-first with legacy fallback).
    backend, precision, which_jn, n_max
        Forwarded to :func:`mie_coefficients`.

    Returns
    -------
    dict
        ``S1``, ``S2``, polarized intensities, and angular metadata.
    """
    # - evaluate mie coefficients (vectorized)
    miecoeff = mie_coefficients(
        k0=k0,
        r_layers=r_layers,
        eps_layers=eps_layers,
        r_c=r_c,
        eps_c=eps_c,
        r_s=r_s,
        eps_s=eps_s,
        eps_env=eps_env,
        backend=backend,
        precision=precision,
        which_jn=which_jn,
        n_max=n_max,
    )
    n_max = miecoeff["n_max"]
    n = miecoeff["n"]
    k = miecoeff["k"]
    k0 = miecoeff["k0"]
    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]

    mu = torch.cos(theta)
    pi_n, tau_n = special.pi_tau(n_max, mu)  # shape: N_teta, n_Mie_order
    pi_n = pi_n[1:]  # Mie orders 1 - n (no zero order)
    tau_n = tau_n[1:]  # Mie orders 1 - n (no zero order)

    # vectorization:
    #   - dim 0: n particles
    #   - dim 1: wavevectors
    #   - dim 2: Mie order
    #   - dim 3: teta angles
    k = k.unsqueeze(0)  # add dim order
    k0 = k0.unsqueeze(0)  # add dim order
    pi_n = pi_n.unsqueeze(1).unsqueeze(1)  # add N_part, k0 dims.
    tau_n = tau_n.unsqueeze(1).unsqueeze(1)  # add N_part, k0 dims.
    n = n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # add dim N_part, N_k0, N_theta
    a_n = a_n.unsqueeze(-1)
    b_n = b_n.unsqueeze(-1)

    # eval. S1 and S2, sum over Mie orders (dim 0)
    s1 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * pi_n + b_n * tau_n), dim=0)
    s2 = torch.sum(((2 * n + 1) / (n * (n + 1))) * (a_n * tau_n + b_n * pi_n), dim=0)

    i_per = s1.abs() ** 2
    i_par = s2.abs() ** 2

    i_unpol = (i_par + i_per) / 2
    pol_degree = (i_per - i_par) / (i_per + i_par)

    return dict(
        wavelength=2 * torch.pi / k0.squeeze(),
        k0=k0.squeeze(),
        theta=theta.squeeze(),
        # observables
        S1=s1,
        S2=s2,
        i_per=i_per,
        i_par=i_par,
        i_unpol=i_unpol,
        pol_degree=pol_degree,
    )


def nearfields(
    k0,
    r_probe,
    r_layers=None,
    eps_layers=None,
    r_c=None,
    eps_c=None,
    r_s=None,
    eps_s=None,
    eps_env=1.0,
    E_0=1,
    backend="pena",
    precision="double",
    which_jn="recurrence",
    n_max=None,
):
    """Compute incident, scattered, and total near fields.

    Parameters
    ----------
    k0 : tensor-like
        Vacuum wavevector(s), rad/nm.
    r_probe : tensor-like
        Cartesian probe coordinates with last dimension 3.
    r_layers, eps_layers, r_c, eps_c, r_s, eps_s, eps_env : tensor-like
        Geometry/material inputs (multilayer-first with legacy fallback).
    E_0 : complex or float, default=1
        Incident field amplitude.
    backend : {"pena", "torch", "scipy"}, default="pena"
        Coefficient backend.
    precision : {"single", "double"}, default="double"
        Precision for torch-based routines.
    which_jn : {"recurrence", "ratios"}, default="recurrence"
        Torch-only spherical ``j_n`` implementation.
    n_max : int, optional
        Truncation order. Auto-estimated when omitted.

    Returns
    -------
    dict
        ``E_i``, ``H_i``, ``E_s``, ``H_s``, ``E_t``, and ``H_t``.
    """
    from pymiediff.special import vsh, vsh_pena
    from pymiediff.helper import transform_xyz_to_spherical
    from pymiediff.helper import transform_fields_spherical_to_cartesian

    backend_l = backend.lower()

    if backend_l == "pena":
        # evaluate scattering coefficients and multilayer metadata
        miecoeff = mie_coefficients(
            k0=k0,
            r_layers=r_layers,
            eps_layers=eps_layers,
            r_c=r_c,
            eps_c=eps_c,
            r_s=r_s,
            eps_s=eps_s,
            eps_env=eps_env,
            backend=backend,
            precision=precision,
            which_jn=which_jn,
            n_max=n_max,
            return_internal=False,
        )

        n = miecoeff["n"]
        n_max = int(miecoeff["n_max"].item())
        k = miecoeff["k"]
        k0 = miecoeff["k0"]
        n_env = miecoeff["n_env"]
        m_layers = miecoeff["m_layers"]  # relative to env, shape (Np, L, Nk)
        r_layers = miecoeff["r_layers"]  # shape (Np, L)
        a_ext = miecoeff["a_n"]  # shape (n_max, Np, Nk)
        b_ext = miecoeff["b_n"]
        eps_layers = miecoeff["eps_layers"]
        L = r_layers.shape[1]

        # prepare full order arrays with n=0 slot (avoid in-place ops for autograd)
        dtype_c = a_ext.dtype
        n_p = r_layers.shape[0]
        n_k0 = k0.shape[1]

        zeros_n = torch.zeros((1, n_p, n_k0), dtype=dtype_c, device=k0.device)
        a_L = torch.cat((zeros_n, a_ext), dim=0)
        b_L = torch.cat((zeros_n, b_ext), dim=0)
        c_L = torch.ones_like(a_L)
        d_L = torch.ones_like(a_L)

        x_layers = k.unsqueeze(1) * r_layers.unsqueeze(-1).to(dtype=dtype_c)  # (Np, L, Nk)
        m_ext = torch.cat(
            (
                m_layers,
                torch.ones((n_p, 1, n_k0), dtype=dtype_c, device=k0.device),
            ),
            dim=1,
        )  # (Np, L+1, Nk)

        eps = torch.tensor(1e-30, dtype=dtype_c, device=k0.device)

        # recursive expansion coefficients, l = L ... 1
        a_layers = [None] * (L + 1)
        b_layers = [None] * (L + 1)
        c_layers = [None] * (L + 1)
        d_layers = [None] * (L + 1)
        a_layers[L] = a_L
        b_layers[L] = b_L
        c_layers[L] = c_L
        d_layers[L] = d_L

        for li in range(L - 1, -1, -1):
            ml = m_ext[:, li, :]
            mlp1 = m_ext[:, li + 1, :]
            xl = x_layers[:, li, :]

            z_l = ml * xl
            z_lp1 = mlp1 * xl

            D1_l = special.pena_D1_n(n_max, z_l, precision=precision)
            D3_l = special.pena_D3_n(n_max, z_l, D1=D1_l, precision=precision)
            psi_l, zeta_l = special.pena_psi_zeta_n(
                n_max, z_l, D1=D1_l, D3=D3_l, precision=precision
            )

            D1_lp1 = special.pena_D1_n(n_max, z_lp1, precision=precision)
            D3_lp1 = special.pena_D3_n(n_max, z_lp1, D1=D1_lp1, precision=precision)
            psi_lp1, zeta_lp1 = special.pena_psi_zeta_n(
                n_max, z_lp1, D1=D1_lp1, D3=D3_lp1, precision=precision
            )

            a_nxt = a_layers[li + 1]
            b_nxt = b_layers[li + 1]
            c_nxt = c_layers[li + 1]
            d_nxt = d_layers[li + 1]

            T1 = a_nxt * zeta_lp1 - d_nxt * psi_lp1
            T2 = b_nxt * zeta_lp1 - c_nxt * psi_lp1
            T3 = d_nxt * D1_lp1 * psi_lp1 - a_nxt * D3_lp1 * zeta_lp1
            T4 = c_nxt * D1_lp1 * psi_lp1 - b_nxt * D3_lp1 * zeta_lp1
            U = D1_l - D3_l

            den_a = zeta_l * U
            den_c = psi_l * U
            den_a = torch.where(den_a.abs() < eps.abs(), den_a + eps, den_a)
            den_c = torch.where(den_c.abs() < eps.abs(), den_c + eps, den_c)

            # Ladutenko et al. 2017, Eq. (7.1)-(7.4): factor m_l / m_{l+1}
            m_ratio = ml / mlp1
            a_l = (D1_l * T1 + T3 * m_ratio) / den_a
            b_l = (D1_l * T2 * m_ratio + T4) / den_a
            c_l = (D3_l * T2 * m_ratio + T4) / den_c
            d_l = (D3_l * T1 + T3 * m_ratio) / den_c

            # enforce regularity in inner core
            if li == 0:
                a_l = torch.zeros_like(a_l)
                b_l = torch.zeros_like(b_l)

            a_layers[li] = a_l
            b_layers[li] = b_l
            c_layers[li] = c_l
            d_layers[li] = d_l

        # stack and remove n=0 slot for field sums
        a_reg = torch.stack(a_layers, dim=-1)[1:, ...]
        b_reg = torch.stack(b_layers, dim=-1)[1:, ...]
        c_reg = torch.stack(c_layers, dim=-1)[1:, ...]
        d_reg = torch.stack(d_layers, dim=-1)[1:, ...]

        # - convert Cartesian to spherical coordinates
        r, theta, phi = transform_xyz_to_spherical(
            r_probe[..., 0], r_probe[..., 1], r_probe[..., 2]
        )

        n_pos = theta.shape[0]
        full_shape = (n_max, n_p, n_k0, n_pos, 3)

        # base tensors
        k = k.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        k0 = k0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        n_env = n_env.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        a_reg = a_reg.unsqueeze(-1).unsqueeze(-1)
        b_reg = b_reg.unsqueeze(-1).unsqueeze(-1)
        c_reg = c_reg.unsqueeze(-1).unsqueeze(-1)
        d_reg = d_reg.unsqueeze(-1).unsqueeze(-1)

        r_sh = r.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        theta_sh = theta.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        phi_sh = phi.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        n = n.view((-1,) + (r_sh.ndim - 1) * (1,))
        En = 1j**n * E_0 * (2 * n + 1) / (n * (n + 1))

        # evaluate VSH per region medium (0..L-1 layers, L outside env)
        n_layers_abs = torch.sqrt(eps_layers)
        n_medium_reg = [
            n_layers_abs[:, li, :].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            for li in range(L)
        ]
        n_medium_reg.append(n_env)

        M1_o, M1_e, N1_o, N1_e = [], [], [], []
        M3_o, M3_e, N3_o, N3_e = [], [], [], []
        for n_med in n_medium_reg:
            _M1o, _M1e, _N1o, _N1e = vsh_pena(
                n_max, k0, n_med, r_sh, theta_sh, phi_sh, kind=1, precision=precision
            )
            _M3o, _M3e, _N3o, _N3e = vsh_pena(
                n_max, k0, n_med, r_sh, theta_sh, phi_sh, kind=3, precision=precision
            )
            M1_o.append(_M1o)
            M1_e.append(_M1e)
            N1_o.append(_N1o)
            N1_e.append(_N1e)
            M3_o.append(_M3o)
            M3_e.append(_M3e)
            N3_o.append(_N3o)
            N3_e.append(_N3e)

        # region masks
        r_pos = r.view(1, 1, n_pos)
        masks = []
        for li in range(L):
            r_hi = r_layers[:, li].view(n_p, 1, 1)
            if li == 0:
                m = r_pos <= r_hi
            else:
                r_lo = r_layers[:, li - 1].view(n_p, 1, 1)
                m = torch.logical_and(r_pos > r_lo, r_pos <= r_hi)
            masks.append(m)
        m_out = r_pos > r_layers[:, -1].view(n_p, 1, 1)
        masks.append(m_out)
        masks = [torch.broadcast_to(m.unsqueeze(0).unsqueeze(-1), full_shape) for m in masks]

        Es = torch.zeros(full_shape, dtype=dtype_c, device=k0.device)
        Hs = torch.zeros_like(Es)

        for li in range(L + 1):
            a_l = a_reg[:, :, :, li, ...]
            b_l = b_reg[:, :, :, li, ...]
            c_l = c_reg[:, :, :, li, ...]
            d_l = d_reg[:, :, :, li, ...]
            n_med = n_medium_reg[li]

            if li == L:
                # outside: scattered field only
                E_l = En * (1j * a_l * N3_e[li] - b_l * M3_o[li])
                H_l = n_med * En * (1j * b_l * N3_o[li] + a_l * M3_e[li])
            else:
                # inside layers: full field expansion (Ladutenko et al. 2017, Eq. 4)
                E_l = En * (
                    c_l * M1_o[li]
                    - 1j * d_l * N1_e[li]
                    + 1j * a_l * N3_e[li]
                    - b_l * M3_o[li]
                )
                H_l = -n_med * En * (
                    d_l * M1_e[li]
                    + 1j * c_l * N1_o[li]
                    - 1j * b_l * N3_o[li]
                    - a_l * M3_e[li]
                )

            Es[masks[li]] = E_l[masks[li]]
            Hs[masks[li]] = H_l[masks[li]]

        # convert to Cartesian and sum over Mie orders
        Es_xyz = transform_fields_spherical_to_cartesian(
            Es[..., 0], Es[..., 1], Es[..., 2], r_sh[..., 0], theta_sh[..., 0], phi_sh[..., 0]
        )
        Es_xyz = torch.stack(Es_xyz, dim=-1).sum(dim=0)

        Hs_xyz = transform_fields_spherical_to_cartesian(
            Hs[..., 0], Hs[..., 1], Hs[..., 2], r_sh[..., 0], theta_sh[..., 0], phi_sh[..., 0]
        )
        Hs_xyz = torch.stack(Hs_xyz, dim=-1).sum(dim=0)

        # incident plane wave in host medium (for API compatibility)
        Ei = torch.zeros_like(Es_xyz)
        Ei[..., 0] = (E_0 * torch.exp(1j * k * r_sh * torch.cos(theta_sh)))[..., 0]

        Hi = torch.zeros_like(Hs_xyz)
        Hi[..., 1] = (E_0 * n_env * torch.exp(1j * k * r_sh * torch.cos(theta_sh)))[..., 0]

        Etot = Es_xyz.clone()
        Htot = Hs_xyz.clone()
        idx_out = masks[-1][0, ...]
        Etot[idx_out] += Ei[idx_out]
        Htot[idx_out] += Hi[idx_out]

        return dict(
            E_i=Ei,
            H_i=Hi,
            E_s=Es_xyz,
            H_s=Hs_xyz,
            E_t=Etot,
            H_t=Htot,
        )

    # - evaluate mie coefficients
    miecoeff = mie_coefficients(
        k0=k0,
        r_layers=r_layers,
        eps_layers=eps_layers,
        r_c=r_c,
        eps_c=eps_c,
        r_s=r_s,
        eps_s=eps_s,
        eps_env=eps_env,
        backend=backend,
        precision=precision,
        which_jn=which_jn,
        n_max=n_max,
        return_internal=True,
    )

    n = miecoeff["n"]
    n_max = miecoeff["n_max"]
    k = miecoeff["k"]
    k0 = miecoeff["k0"]
    r_c = miecoeff["r_c"]
    r_s = miecoeff["r_s"]

    n_env = miecoeff["n_env"]
    n_sourrounding = n_env
    n_c = miecoeff["n_c"]
    n_core = n_c
    n_s = miecoeff["n_s"]
    n_shell = n_s

    a_n = miecoeff["a_n"]
    b_n = miecoeff["b_n"]
    c_n = miecoeff["c_n"]
    d_n = miecoeff["d_n"]
    f_n = miecoeff["f_n"]
    g_n = miecoeff["g_n"]
    v_n = miecoeff["v_n"]
    w_n = miecoeff["w_n"]

    kc = miecoeff["k0"] * r_c
    ks = miecoeff["k0"] * r_s

    # - convert Cartesian to spherical coordinates
    r, theta, phi = transform_xyz_to_spherical(
        r_probe[..., 0], r_probe[..., 1], r_probe[..., 2]
    )

    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)
    assert n_max >= 0

    # vectorization:
    #   - dim 0: Mie order
    #   - dim 1: n particles
    #   - dim 2: wavevectors
    #   - dim 3: positions
    #   - dim 4: field vector components (3)
    n_p = r_c.shape[0]
    n_k0 = k0.shape[1]
    n_pos = theta.shape[0]
    full_shape = (n_max, n_p, n_k0, n_pos, 3)

    # expand dimensions
    # add order, position, vector dim
    k = k.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    k0 = k0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    kc = kc.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    ks = ks.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    n_sourrounding = n_sourrounding.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # add position, vector dim
    a_n = a_n.unsqueeze(-1).unsqueeze(-1)
    b_n = b_n.unsqueeze(-1).unsqueeze(-1)
    c_n = c_n.unsqueeze(-1).unsqueeze(-1)
    d_n = d_n.unsqueeze(-1).unsqueeze(-1)
    f_n = f_n.unsqueeze(-1).unsqueeze(-1)
    g_n = g_n.unsqueeze(-1).unsqueeze(-1)
    v_n = v_n.unsqueeze(-1).unsqueeze(-1)
    w_n = w_n.unsqueeze(-1).unsqueeze(-1)

    # add order, particle, wavenumber, vector dimensions
    r = r.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    idx_1 = r <= r_c  # positions in core
    idx_2 = torch.logical_and(r_c < r, r <= r_s)  # positions in core
    idx_3 = r > r_s  # outside positions

    phi = phi.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    theta = theta.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    # add particle, wavenumber, position, vector dimensions
    n = n.view((-1,) + (r.ndim - 1) * (1,))

    # evaluate vector spherical harmonics
    # Note: this is not optimum as all VSH are evaluated for all positions
    # TODO: check positions first (inside core, inside shell, outside)
    M1_o1n_c, M1_e1n_c, N1_o1n_c, N1_e1n_c = vsh(
        n_max, k0, n_core, r, theta, phi, kind=1
    )
    M1_o1n_s, M1_e1n_s, N1_o1n_s, N1_e1n_s = vsh(
        n_max, k0, n_shell, r, theta, phi, kind=1
    )
    M2_o1n_s, M2_e1n_s, N2_o1n_s, N2_e1n_s = vsh(
        n_max, k0, n_shell, r, theta, phi, kind=2
    )
    M3_o1n, M3_e1n, N3_o1n, N3_e1n = vsh(
        n_max, k0, n_sourrounding, r, theta, phi, kind=3
    )

    # - scattered fields (Bohren Huffmann, Eq. 4.40, 4.45, 8.0)
    # with En = i^n E0 (2n+1)/(n(n+1)):
    # Es = sum_n En (i a_n N3e1n - b_n M3o1n)
    # Hs = k/(omega mu) sum_n En (i b_n N3o1n + a_n M3e1n)
    # the resulting fields are the spherical coordinate components

    En = 1j**n * E_0 * (2 * n + 1) / (n * (n + 1))
    idx_1 = torch.broadcast_to(idx_1, full_shape)
    idx_2 = torch.broadcast_to(idx_2, full_shape)
    idx_3 = torch.broadcast_to(idx_3, full_shape)

    # electric fields (relative to E0)
    Es_1 = En * (c_n * M1_o1n_c - 1j * d_n * N1_e1n_c)
    Es_2 = En * (
        (f_n * M1_o1n_s - 1j * g_n * N1_e1n_s) - (v_n * M2_o1n_s - 1j * w_n * N2_e1n_s)
    )
    Es_3 = En * (1j * a_n * N3_e1n - b_n * M3_o1n)

    Es = torch.zeros(full_shape, dtype=a_n.dtype, device=a_n.device)
    Es[idx_1] = Es_1[idx_1]
    Es[idx_2] = Es_2[idx_2]
    Es[idx_3] = Es_3[idx_3]

    # magnetic fields (relative to H0)
    Hs_1 = -n_core * En * (d_n * M1_e1n_c + 1j * c_n * N1_o1n_c)
    Hs_2 = (
        -n_shell
        
        * En
        * (
            (g_n * M1_e1n_s + 1j * f_n * N1_o1n_s)
            - (w_n * M2_e1n_s + 1j * v_n * N2_o1n_s)
        )
    )
    Hs_3 = En*n_env * (1j * b_n * N3_o1n + a_n * M3_e1n)

    Hs = torch.zeros(full_shape, dtype=a_n.dtype, device=a_n.device)
    Hs[idx_1] = Hs_1[idx_1]
    Hs[idx_2] = Hs_2[idx_2]
    Hs[idx_3] = Hs_3[idx_3]

    # convert to Cartesian
    Es_xyz = transform_fields_spherical_to_cartesian(
        Es[..., 0], Es[..., 1], Es[..., 2], r[..., 0], theta[..., 0], phi[..., 0]
    )
    Es_xyz = torch.stack(Es_xyz, dim=-1)

    Hs_xyz = transform_fields_spherical_to_cartesian(
        Hs[..., 0], Hs[..., 1], Hs[..., 2], r[..., 0], theta[..., 0], phi[..., 0]
    )
    Hs_xyz = torch.stack(Hs_xyz, dim=-1)

    # sum Mie orders
    Es_xyz = Es_xyz.sum(dim=0)
    Hs_xyz = Hs_xyz.sum(dim=0)

    # incident field: X-pol. plane wave
    # expansion (B&H Eq. 4.37)
    # E_pw = E0 * sum_i [ i^n * (2n+1) / (n(n+1)) * ( M1_o1n - i * N1_e1n ) ]
    # H_pw = (E0 / eta) * sum_i [ i^n * (2n+1) / (n(n+1)) * ( M1_e1n + i * N1_o1n ) ]
    # E0 = En * (M1_o1n_c - 1j * N1_e1n_c)
    # H0 = En * (M1_e1n_c + 1j * N1_o1n_c)
    Ei = torch.zeros_like(Es_xyz)
    Ei[..., 0] = (E_0 * torch.exp(1j * k * r * torch.cos(theta)))[..., 0]

    Hi = torch.zeros_like(Es_xyz)
    Hi[..., 1] = (E_0*n_env * torch.exp(1j * k * r * torch.cos(theta)))[..., 0]

    # add incident field to outside positions
    Etot = Es_xyz.clone()
    Htot = Hs_xyz.clone()
    Etot[idx_3[0, ...]] += Ei[idx_3[0, ...]]
    Htot[idx_3[0, ...]] += Hi[idx_3[0, ...]]

    return_dict = dict(
        E_i=Ei,
        H_i=Hi,
        E_s=Es_xyz,
        H_s=Hs_xyz,
        E_t=Etot,
        H_t=Htot,
    )
    
    return return_dict
