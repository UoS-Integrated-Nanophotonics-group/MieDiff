# -*- coding: utf-8 -*-
"""
auto-diff ready wrapper of scipy spherical Bessel functions
"""
# %%
import warnings

import torch
from scipy.special import spherical_jn, spherical_yn
import numpy as np


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


# %% ---- testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_pt_test = 50
    N_order_test = 1

    n = torch.tensor(5)
    # z = torch.rand(3, dtype=torch.complex64).unsqueeze(0)
    z = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)

    z.requires_grad = True
    result = Jn(n, z)

    # numerical center diff. for testing:
    def num_dJn_dz(n, z, eps=0.0001 + 0.0001j):
        z = z.conj()
        fm = Jn(n, z - eps)
        fp = Jn(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz

    numdz = num_dJn_dz(n, z)
    numdz_np = numdz.detach().numpy().squeeze()

    # batch eval does seem to work only for one dimension of outputs
    # for (N,M) inputs, the sum of gradients along N seems to be calculated...
    z_grad = torch.autograd.grad(
        outputs=result,
        inputs=[z],
        grad_outputs=torch.ones_like(result),
    )

    z_np = z.detach().numpy().squeeze()
    res_np = result.detach().numpy().squeeze()
    z_grad_plot = z_grad[0].detach().numpy().squeeze()

    plt.subplot(211, title="real")
    plt.plot(z_np, res_np.real, label="fwd")
    plt.plot(z_np, numdz_np.real, label="num. grad.")
    plt.plot(z_np, z_grad_plot.real, label="AD grad.", dashes=[2, 2])
    plt.legend()

    plt.subplot(212, title="imag")
    plt.plot(z_np, res_np.imag, label="fwd")
    plt.plot(z_np, numdz_np.imag, label="num. grad.")
    plt.plot(z_np, z_grad_plot.imag, label="AD grad.", dashes=[2, 2])

    plt.tight_layout()
    plt.show()

    # caution with complex arguments:
    # pytorch uses internally the convention that complex derivatives are wrt the conjugate:
    # derivative = dF/dz*. 
    # Therefore, there is a sign difference at the imag part of the output, which needs 
    # to be accounted for in the backward pass implementation.
    # see: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers

    torch.autograd.gradcheck(Jn, (n, z), eps=0.01)
