# -*- coding: utf-8 -*-
"""General-purpose helper utilities used across :mod:`pymiediff`.

The functions in this module support three main tasks:

1. tensor post-processing for user-facing APIs,
2. truncation-order estimation for Mie series,
3. small math utilities for coordinates, interpolation, and testing.

Most helpers are thin and side-effect free. They are written to accept tensor
inputs directly and preserve PyTorch dtypes/devices where possible.

Examples
--------
>>> import torch
>>> from pymiediff.helper.helper import get_truncution_criteroin_wiscombe
>>> get_truncution_criteroin_wiscombe(torch.tensor([3.0]))
10
"""
import torch


def detach_tensor(args, item=False):
    """Detach tensors and convert them to NumPy arrays.

    Parameters
    ----------
    args : torch.Tensor or tuple of torch.Tensor
        Tensor object(s) to convert.
    item : bool, default=False
        If ``True`` and ``args`` is a tuple, each converted array is reduced
        with ``.item()``.

    Returns
    -------
    numpy.ndarray or tuple
        Converted object with the same container structure as ``args``.
    """
    # If args is a tuple, process its elements; otherwise, process the single tensor
    if isinstance(args, tuple) and not item:
        return tuple(x.detach().numpy() for x in args)
    elif isinstance(args, tuple) and item:
        return tuple(x.detach().numpy().item() for x in args)
    else:
        return args.detach().numpy()


def get_truncution_criteroin_wiscombe(ka):
    """Estimate ``n_max`` with Wiscombe's truncation criterion.

    Parameters
    ----------
    ka : torch.Tensor or array-like
        Size parameter(s) ``k * r_outer``.

    Returns
    -------
    int
        Recommended maximum multipole order.

    Notes
    -----
    Uses the piecewise empirical rule from Wiscombe, *Appl. Opt.* 19, 1505
    (1980).
    """
    ka = torch.max(torch.abs(ka))

    if ka <= 8:
        n_max = int(torch.round(1 + ka + 4.0 * (ka ** (1 / 3))))
    elif 8 < ka < 4200:
        n_max = int(torch.round(2 + ka + 4.05 * (ka ** (1 / 3))))
    else:
        n_max = int(torch.round(2 + ka + 4.0 * (ka ** (1 / 3))))

    return n_max


def get_truncution_criteroin_pena2009(k0, r_layers, eps_layers, eps_env):
    """Peña/Pal (2009) truncation criterion for multilayer spheres.

    Parameters
    ----------
    k0 : torch.Tensor or array-like
        Vacuum wavevector(s), typically shaped ``(1, N_k0)`` after broadcasting.
    r_layers : torch.Tensor or array-like
        Layer outer radii with shape ``(N_part, L)``.
    eps_layers : torch.Tensor or array-like
        Layer permittivities with shape ``(N_part, L, N_k0)``.
    eps_env : torch.Tensor or array-like
        Environment permittivity with shape ``(1, N_k0)``.

    Returns
    -------
    int
        Recommended maximum multipole order.

    Notes
    -----
    Uses
    ``Nmax = max_l(max(Nstop, |m_l x_l|, |m_l x_{l-1}|)) + 15``
    with host-medium size parameters.
    """
    k0 = torch.as_tensor(k0)
    r_layers = torch.as_tensor(r_layers)
    eps_layers = torch.as_tensor(eps_layers)
    eps_env = torch.as_tensor(eps_env)

    # expected broadcast-ready shapes:
    #   k0:        (1, N_k0)
    #   r_layers:  (N_part, L)
    #   eps_layers:(N_part, L, N_k0)
    #   eps_env:   (1, N_k0)
    if r_layers.ndim != 2:
        raise ValueError("`r_layers` must have shape (N_part, L).")
    if eps_layers.ndim != 3:
        raise ValueError("`eps_layers` must have shape (N_part, L, N_k0).")

    n_env = torch.sqrt(eps_env)
    n_layers = torch.sqrt(eps_layers)

    # x_l = k * r_l with k in host medium
    x_layers = k0.unsqueeze(1) * n_env.unsqueeze(1) * r_layers.unsqueeze(-1)
    x_outer = torch.abs(x_layers[:, -1, :]).max()

    # N_stop (Wiscombe piecewise in terms of outer size parameter x_L)
    if x_outer <= 8:
        n_stop = torch.round(x_outer + 4.0 * (x_outer ** (1 / 3)) + 1.0)
    elif x_outer < 4200:
        n_stop = torch.round(x_outer + 4.05 * (x_outer ** (1 / 3)) + 2.0)
    else:
        n_stop = torch.round(x_outer + 4.0 * (x_outer ** (1 / 3)) + 2.0)

    # |m_l x_l| = |k0 * n_l * r_l|
    m_x_l = torch.abs(k0.unsqueeze(1) * n_layers * r_layers.unsqueeze(-1))

    # |m_l x_{l-1}|, with x_0 = 0 for the first layer
    r_prev = torch.cat(
        (torch.zeros_like(r_layers[:, :1]), r_layers[:, :-1]),
        dim=1,
    )
    m_x_prev = torch.abs(k0.unsqueeze(1) * n_layers * r_prev.unsqueeze(-1))

    n_raw = torch.maximum(n_stop, torch.maximum(m_x_l.max(), m_x_prev.max()))
    n_max = int(torch.round(n_raw + 15.0).item())
    return n_max


# --- plane wave VSH expansion
def plane_wave_expansion(n):
    """Return plane-wave VSH expansion coefficient ``a_pw_n``.

    Parameters
    ----------
    n : int or torch.Tensor
        Maximum expansion order.

    Returns
    -------
    dict
        Dictionary containing ``a_pw_n`` with coefficients for orders
        ``1..n``.
    """
    # canonicalize n_max
    if isinstance(n, torch.Tensor):
        n_max = int(n.max().item())
    else:
        n_max = int(n)
    assert n_max >= 0

    n_all = torch.arange(1, n_max + 1)

    a_pw_n = 1j**n_all * (2 * n_all + 1) / (n_all * (n_all + 1))

    return dict(a_pw_n=a_pw_n)


# --- Cartesian <--> spherical
def transform_fields_spherical_to_cartesian(E_r, E_t, E_ph, r, theta, phi):
    """Convert field components from spherical to Cartesian basis.

    Parameters
    ----------
    E_r, E_t, E_ph : torch.Tensor
        Spherical vector components.
    r : torch.Tensor
        Radius values (not used directly; kept for API symmetry).
    theta : torch.Tensor
        Polar angle from the positive ``z`` axis.
    phi : torch.Tensor
        Azimuth angle from the positive ``x`` axis.

    Returns
    -------
    tuple of torch.Tensor
        Cartesian components ``(E_x, E_y, E_z)``.
    """
    # spherical unit vectors in cartesian basis in terms of theta (t) and phi (p):
    # e_r  = [e_x sin t cos p, e_y sin t sin p, e_z cos t]
    # e_th = [e_x cos t cos p, e_y cos t sin p, -e_z sin t]
    # e_ph = [-e_x sin p,      e_y cos p,       0]
    st = torch.sin(theta)
    ct = torch.cos(theta)
    sp = torch.sin(phi)
    cp = torch.cos(phi)

    Ex = E_r * (st * cp) + E_t * (ct * cp) + E_ph * (-sp)
    Ey = E_r * (st * sp) + E_t * (ct * sp) + E_ph * (cp)
    Ez = E_r * (ct) + E_t * (-st) + E_ph * (torch.zeros_like(phi))

    return Ex, Ey, Ez


def transform_spherical_to_xyz(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    r : torch.Tensor
        Radius.
    theta : torch.Tensor
        Polar angle.
    phi : torch.Tensor
        Azimuth angle.

    Returns
    -------
    tuple of torch.Tensor
        Cartesian coordinates ``(x, y, z)``.
    """
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z


def transform_xyz_to_spherical(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : torch.Tensor
        Cartesian coordinates.

    Returns
    -------
    tuple of torch.Tensor
        Spherical coordinates ``(r, theta, phi)``.
    """
    r = torch.sqrt(x**2 + y**2 + z**2)
    r_safe = torch.where(r == 0, torch.as_tensor(1e-12, device=r.device, dtype=r.dtype), r)
    z_over_r = z / r_safe
    eps = torch.finfo(z_over_r.dtype).eps * 10
    z_over_r = torch.clamp(z_over_r, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(z_over_r)
    eps_phi = torch.finfo(x.dtype).eps * 10
    x_safe = x + (x == 0).to(x.dtype) * eps_phi
    phi = torch.atan2(y, x_safe)
    return r, theta, phi


# numerical center diff. for testing:
def num_center_diff(Funct, n, z, eps=0.0001 + 0.0001j):
    """Compute a complex-valued central finite-difference derivative.

    Parameters
    ----------
    Funct : callable
        Function called as ``Funct(n, z)``.
    n : Any
        Additional argument passed to ``Funct``.
    z : torch.Tensor
        Evaluation point.
    eps : complex, default=0.0001+0.0001j
        Finite-difference step.

    Returns
    -------
    torch.Tensor
        Numerical derivative with respect to ``z``.
    """
    z = z.conj()
    fm = Funct(n, z - eps)
    fp = Funct(n, z + eps)
    dz = (fp - fm) / (2 * eps)
    return dz


def funct_grad_checker(z, funct, inputs):
    """Compare autograd and numerical derivatives.

    Parameters
    ----------
    z : torch.Tensor
        Tensor with ``requires_grad=True`` used as differentiation variable.
    funct : callable
        Function to evaluate.
    inputs : tuple
        Positional inputs passed to ``funct`` and numerical differentiation.

    Returns
    -------
    tuple
        ``(z_np, num_grad_np, grad_np)`` as NumPy arrays.
    """
    result = funct(*inputs)
    num_grad = num_center_diff(funct, *inputs)
    grad = torch.autograd.grad(
        outputs=result, inputs=[z], grad_outputs=torch.ones_like(result)
    )

    z_np = z.detach().numpy().squeeze()
    num_grad_np = num_grad.detach().numpy().squeeze()
    grad_np = grad[0].detach().numpy().squeeze()

    return z_np, num_grad_np, grad_np


def interp1d(x_eval: torch.Tensor, x_dat: torch.Tensor, y_dat: torch.Tensor):
    """One-dimensional linear interpolation in PyTorch.

    Parameters
    ----------
    x_eval : torch.Tensor
        Coordinates where interpolation is evaluated.
    x_dat : torch.Tensor
        Sample coordinates.
    y_dat : torch.Tensor
        Sample values, same length as ``x_dat``.

    Returns
    -------
    torch.Tensor
        Interpolated values with shape of ``x_eval``.

    Examples
    --------
    >>> import torch
    >>> interp1d(torch.tensor([0.5]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 2.0]))
    tensor([1.])
    """
    assert len(x_dat) == len(y_dat)
    assert not torch.is_complex(x_dat)

    # sort x input data
    i_sort = torch.argsort(x_dat)
    _x = x_dat[i_sort]
    _y = y_dat[i_sort]

    # find left/right neighbor x datapoints
    idx_r = torch.bucketize(x_eval, _x)
    idx_l = idx_r - 1
    idx_r = idx_r.clamp(0, _x.shape[0] - 1)
    idx_l = idx_l.clamp(0, _x.shape[0] - 1)

    # distances to left / right (=weights)
    dist_l = x_eval - _x[idx_l]
    dist_r = _x[idx_r] - x_eval
    dist_l[dist_l < 0] = 0.0
    dist_r[dist_r < 0] = 0.0
    dist_l[torch.logical_and(dist_l == 0, dist_r == 0)] = 1.0
    sum_d_l_r = dist_l + dist_r
    y_l = _y[idx_l]
    y_r = _y[idx_r]

    # bilinear interpolated values
    y_eval = (y_l * dist_r + y_r * dist_l) / sum_d_l_r

    return y_eval


def _squeeze_dimensions(results_dict):
    """Squeeze singleton dimensions for all tensor values in a dictionary.

    Parameters
    ----------
    results_dict : dict
        Mapping whose tensor values are squeezed in-place.

    Returns
    -------
    dict
        The same dictionary instance after in-place squeezing.
    """
    for k in results_dict:
        if type(results_dict[k]) == torch.Tensor:
            results_dict[k] = results_dict[k].squeeze()
    return results_dict
