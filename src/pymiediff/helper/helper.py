import numpy as np
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
    ka = np.max(ka)

    if ka <= 8:
        n_max = int(np.round(1 + ka + 4.0 * (ka ** (1 / 3))))
    elif 8 < ka < 4200:
        n_max = int(np.round(2 + ka + 4.05 * (ka ** (1 / 3))))
    else:
        n_max = int(np.round(2 + ka + 4.0 * (ka ** (1 / 3))))

    return n_max


def make_multipoles(As, Bs, k):
    multipoles = []
    for n, (a, b) in enumerate(zip(As, Bs)):
        multipoles.append(
            [
                (2 * np.pi / k.detach().numpy() ** 2)
                * (2 * (n + 1) + 1)
                * (a.abs() ** 2).detach().numpy(),
                (2 * np.pi / k.detach().numpy() ** 2)
                * (2 * (n + 1) + 1)
                * (b.abs() ** 2).detach().numpy(),
            ]
        )
    return multipoles


# numerical center diff. for testing:
def num_center_diff(Funct, n, z, eps=0.0001 + 0.0001j):
    z = z.conj()
    fm = Funct(n, z - eps)
    fp = Funct(n, z + eps)
    dz = (fp - fm) / (2 * eps)
    return dz


def funct_grad_checker(z, funct, inputs, ax=None, check=None, imag=True):
    result = funct(*inputs)
    num_grad = num_center_diff(funct, *inputs)
    grad = torch.autograd.grad(
        outputs=result, inputs=[z], grad_outputs=torch.ones_like(result)
    )

    result_np = result.detach().numpy().squeeze()
    z_np = z.detach().numpy().squeeze()
    num_grad_np = num_grad.detach().numpy().squeeze()
    grad_np = grad[0].detach().numpy().squeeze()

    if ax is not None:
        from pymiediff.helper.plotting import plot_grad_checker
        plot_grad_checker(
            ax[0],
            ax[1],
            z_np,
            result_np,
            grad_np,
            num_grad_np,
            funct.__name__,
            check=check,
            imag=imag,
        )

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
