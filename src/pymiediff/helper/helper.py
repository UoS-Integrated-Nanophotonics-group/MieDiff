# -*- coding: utf-8 -*-
"""pymiediff.helper – Utility functions for Mie‑scattering calculations.

This submodule groups a collection of small, self‑contained helpers that are
used throughout the :pymiediff: package.  All functions operate on
`torch.Tensor` objects (real or complex) and are deliberately written to be
compatible with PyTorch’s autograd system.

The utilities can be grouped into the following categories:

* **Tensor handling**
  - :func:`detach_tensor` – Convert one or several tensors to NumPy while
    optionally extracting a scalar ``.item()``.
* **Series‑truncation criteria**
  - :func:`get_truncution_criteroin_wiscombe` – Wiscombe’s empirical rule for
    choosing the maximum order ``n_max`` of the far‑field Mie series.
* **Vector‑spherical‑harmonic (VSH) expansions**
  - :func:`plane_wave_expansion` – Returns the VSH expansion coefficients for
    a plane wave up to a given order.
* **Coordinate transformations**
  - :func:`transform_fields_spherical_to_cartesian` – Convert field components
    from spherical to Cartesian basis.
  - :func:`transform_spherical_to_xyz` / :func:`transform_xyz_to_spherical` – Pure
    coordinate conversions.
* **Numerical utilities**
  - :func:`num_center_diff` – Central finite‑difference derivative (complex step).
  - :func:`funct_grad_checker` – Compare analytic autograd gradients with the
    numerical derivative.
  - :func:`interp1d` – Simple 1‑D linear interpolation implemented with PyTorch.
  - :func:`_squeeze_dimensions` – Helper to remove singleton dimensions from a
    results dictionary.

All functions assume inputs are `torch.Tensor` objects and will raise
`AssertionError` if the expectations are not met (e.g. mismatched lengths,
complex‑valued ``x_dat`` for ``interp1d``).  The module does not have any
external side effects and can be imported safely in any environment where
PyTorch is available.
"""
import torch


def detach_tensor(args, item=False):
    # If args is a tuple, process its elements; otherwise, process the single tensor
    if isinstance(args, tuple) and not item:
        return tuple(x.detach().numpy() for x in args)
    elif isinstance(args, tuple) and item:
        return tuple(x.detach().numpy().item() for x in args)
    else:
        return args.detach().numpy()


def get_truncution_criteroin_wiscombe(ka):
    # criterion for farfield series truncation for ka = k * r_outer
    #
    # Wiscombe, W. J.
    # "Improved Mie scattering algorithms."
    # Appl. Opt. 19.9, 1505–1509 (1980)
    #
    ka = torch.max(torch.abs(ka))

    if ka <= 8:
        n_max = int(torch.round(1 + ka + 4.0 * (ka ** (1 / 3))))
    elif 8 < ka < 4200:
        n_max = int(torch.round(2 + ka + 4.05 * (ka ** (1 / 3))))
    else:
        n_max = int(torch.round(2 + ka + 4.0 * (ka ** (1 / 3))))

    return n_max


# --- plane wave VSH expansion
def plane_wave_expansion(n):
    """VSH plane wave expansion coefficients up to order n"""
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
    """
    Convert spherical vector components (E_r, E_theta, E_phi) into Cartesian components (Ex,Ey,Ez).

    theta: polar angle from +z axis
    phi: azimuthal angle from +x axis.

    Returns Ex, Ey, Ez (torch complex tensors).
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
    """Transform spherical to cartesian coordinates

    Args:
        r (torch.Tensor): radii
        theta (torch.Tensor): polar angles
        phi (torch.Tensor): azimuth angles

    Returns:
        tuple of torch.Tensor: x, y, z
    """
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z


def transform_xyz_to_spherical(x, y, z):
    """Transform cartesian to spherical coordinates

    Args:
        x (torch.Tensor): x-value of coordinates
        y (torch.Tensor): y-value of coordinates
        z (torch.Tensor): z-value of coordinates

    Returns:
        tuple: r, theta and phi
    """
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    return r, theta, phi


# numerical center diff. for testing:
def num_center_diff(Funct, n, z, eps=0.0001 + 0.0001j):
    z = z.conj()
    fm = Funct(n, z - eps)
    fp = Funct(n, z + eps)
    dz = (fp - fm) / (2 * eps)
    return dz


def funct_grad_checker(z, funct, inputs):
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
    """1D bilinear interpolation

    simple torch implementation of :func:`numpy.interp`

    Args:
        x_eval (torch.Tensor): The x-coordinates at which to evaluate the interpolated values.
        x_dat (torch.Tensor): The x-coordinates of the data points
        y_dat (torch.Tensor): The y-coordinates of the data points, same length as `x_dat`.

    Returns:
        torch.Tensor: The interpolated values, same shape as `x_eval`
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
    for k in results_dict:
        if type(results_dict[k]) == torch.Tensor:
            results_dict[k] = results_dict[k].squeeze()
    return results_dict
