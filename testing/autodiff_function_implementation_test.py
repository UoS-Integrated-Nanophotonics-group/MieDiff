# -*- coding: utf-8 -*-
"""
test script to understand torch autograd function 
implementation, in particular how to do the backward pass
"""
# %%
import warnings

import torch
import numpy as np


class AutoDiffSin(torch.autograd.Function):
    @staticmethod
    def forward(n, z):
        result = torch.sin(n * z)
        
        # calc. the derivatives here for 2nd order diff support.
        # see: https://pytorch.org/docs/stable/notes/extending.html#example
        dn = z * torch.cos(n * z)
        dz = n * torch.cos(n * z)

        return result, dn, dz

    @staticmethod
    def setup_context(ctx, inputs, output):
        n, z = inputs
        result, dn, dz = output
        ctx.save_for_backward(n, z, result, dn, dz)

    @staticmethod
    def backward(ctx, grad_result, grad_dn, grad_dz):
        n, z, result, dn, dz = ctx.saved_tensors

        # chain rule (second term: double diff)
        # differentiation wrt `z`
        ddz = -1 * n**2 * torch.sin(n * z)
        grad_wrt_z = grad_result * dz + grad_dz * ddz

        # differentiation wrt `n`
        ddn = -1 * z**2 * torch.sin(n * z)
        grad_wrt_n = grad_result * dn + grad_dn * ddn

        # return a gradient tensor for each input of "forward"
        # (--> 2: one for n, one for z)
        return grad_wrt_n, grad_wrt_z


# wrapper function
def sin_own(n, z):
    n = torch.as_tensor(n, dtype=torch.float64)
    z = torch.as_tensor(z, dtype=torch.float64)
    result, _dn, _dz = AutoDiffSin.apply(n, z)
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_pt_test = 100
    N_order_test = 1

    n = torch.tensor(6.0)
    z = torch.linspace(1, 5, N_pt_test)

    z.requires_grad = True
    n.requires_grad = True
    result = 0.5 * z + sin_own(n, z) - z * sin_own(n, z) / 5

    # batch eval does seem to work only for one dimension of outputs
    # for (N,M) inputs, the sum of gradients along N seems to be calculated...
    z_grad = torch.autograd.grad(
        outputs=result,
        inputs=[z],
        grad_outputs=torch.ones_like(result),
    )
    print(len(z_grad))
    print(result.shape)
    print(z_grad[0].shape)

    n_np = n.detach().numpy().squeeze()
    z_np = z.detach().numpy().squeeze()
    res_np = result.detach().numpy().squeeze()
    z_grad_plot = z_grad[0].detach().numpy().squeeze()

    plt.plot(z_np, res_np, label="fwd")
    plt.plot(z_np, np.gradient(res_np, z_np), label="num. grad.")
    plt.plot(z_np, z_grad_plot, label="AD grad.", dashes=[2, 2])
    plt.legend()

    plt.tight_layout()
    plt.show()

    torch.autograd.gradcheck(sin_own, [n, z], eps=0.01)

