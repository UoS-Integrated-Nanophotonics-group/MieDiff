# -*- coding: utf-8 -*-
"""
auto-diff ready wrapper of scipy spherical Bessel functions
"""
# %%
import warnings

import torch
from scipy.special import spherical_jn, spherical_yn
# import numpy as np


def bessel2ndDer(n, z, bessel):
    return (1/z**2)*((n**2 - n - z**2) * bessel(n, z) + 2*z*bessel(n + 1, z))


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
def dJn(n: torch.Tensor, z: torch.Tensor):
    """derivative of spherical Bessel function of first kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffdJn.apply(n, z)
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
def Yn(n: torch.Tensor, z: torch.Tensor):
    """spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffYn.apply(n, z)
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
def dYn(n: torch.Tensor, z: torch.Tensor):
    """derivative of spherical Bessel function of second kind

    Args:
        n (torch.Tensor): integer order
        z (torch.Tensor): complex argument

    Returns:
        torch.Tensor: result
    """
    n = torch.as_tensor(n, dtype=torch.int)
    z = torch.as_tensor(z)
    result = _AutoDiffdYn.apply(n, z)
    return result

def sph_h1n(z, n):
    return Jn(n, z) + 1j*Yn(n, z)


def sph_h1n_der(z, n):
    return dJn(n, z) + 1j*dYn(n, z)


def psi(z, n):
    return z*Jn(n, z)


def chi(z, n):
    return -z*Yn(n, z)


def xi(z, n):
    return z*sph_h1n(z, n)


def psi_der(z, n):
    return Jn(n, z) + z*dJn(n, z)


def chi_der(z, n):
    return -Yn(n, z) - z*dYn(n, z)


def xi_der(z, n):
    return sph_h1n(z, n) + z*sph_h1n_der(z, n)




# %% ---- testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N_pt_test = 50
    N_order_test = 1

    n = torch.tensor(5.0)
    # z = torch.rand(3, dtype=torch.complex64).unsqueeze(0)
    # Jn
    z1 = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z1.requires_grad = True
    # Yn
    z2 = torch.linspace(0.5, 5, N_pt_test) + 1j * torch.linspace(1, 5, N_pt_test)
    z2.requires_grad = True
    # dJn
    z3 = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z3.requires_grad = True
    # dYn
    z4 = torch.linspace(0.5, 5, N_pt_test) + 1j * torch.linspace(1, 5, N_pt_test)
    z4.requires_grad = True


    result_Jn = Jn(n, z1)
    result_Yn = Yn(n, z2)
    result_dJn = dJn(n, z3)
    result_dYn = dYn(n, z4)


    # numerical center diff. for testing:
    def num_dJn_dz(n, z, eps=0.0001 + 0.0001j):
        z = z.conj()
        fm = Jn(n, z - eps)
        fp = Jn(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz

    def num_dYn_dz(n, z, eps=0.0001 + 0.0001j):
        z = z.conj()
        fm = Yn(n, z - eps)
        fp = Yn(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz

    def num_ddJn_ddz(n, z, eps=0.0001 + 0.0001j):
        z = z.conj()
        fm = dJn(n, z - eps)
        fp = dJn(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz


    def num_ddYn_ddz(n, z, eps=0.0001 + 0.0001j):
        z = z.conj()
        fm = dYn(n, z - eps)
        fp = dYn(n, z + eps)
        dz = (fp - fm) / (2 * eps)
        return dz

    numdz_dJ = num_dJn_dz(n, z1)
    numdz_dY = num_dYn_dz(n, z2)
    numdz_ddJ = num_ddJn_ddz(n, z3)
    numdz_ddY = num_ddYn_ddz(n, z4)

    numdz_dJ_np = numdz_dJ.detach().numpy().squeeze()
    numdz_dY_np = numdz_dY.detach().numpy().squeeze()
    numdz_ddJ_np = numdz_ddJ.detach().numpy().squeeze()
    numdz_ddY_np = numdz_ddY.detach().numpy().squeeze()

    # batch eval does seem to work only for one dimension of outputs
    # for (N,M) inputs, the sum of gradients along N seems to be calculated...
    z_grad_Jn = torch.autograd.grad(
        outputs=result_Jn,
        inputs=[z1],
        grad_outputs=torch.ones_like(result_Jn)
    )
    z_grad_Yn = torch.autograd.grad(
        outputs=result_Yn,
        inputs=[z2],
        grad_outputs=torch.ones_like(result_Yn)
    )
    z_grad_dJn = torch.autograd.grad(
        outputs=result_dJn,
        inputs=[z3],
        grad_outputs=torch.ones_like(result_dJn)
    )
    z_grad_dYn = torch.autograd.grad(
        outputs=result_dYn,
        inputs=[z4],
        grad_outputs=torch.ones_like(result_dYn)
    )

    z_np = z1.detach().numpy().squeeze()

    z_np_y = z2.detach().numpy().squeeze()

    res_Jn_np = result_Jn.detach().numpy().squeeze()
    res_Yn_np = result_Yn.detach().numpy().squeeze()
    res_dJn_np = result_dJn.detach().numpy().squeeze()
    res_dYn_np = result_dYn.detach().numpy().squeeze()

    z_grad_plolt_Jn = z_grad_Jn[0].detach().numpy().squeeze()
    z_grad_plolt_Yn = z_grad_Yn[0].detach().numpy().squeeze()
    z_grad_plolt_dJn = z_grad_dJn[0].detach().numpy().squeeze()
    z_grad_plolt_dYn = z_grad_dYn[0].detach().numpy().squeeze()


    # Jn
    plt.subplot(421, title="real Jn")
    plt.plot(z_np, res_Jn_np.real, label="fwd")
    plt.plot(z_np, numdz_dJ_np.real, label="num. grad.")
    plt.plot(z_np, z_grad_plolt_Jn.real, label="AD grad.", dashes=[2, 2])
    plt.legend()

    plt.subplot(422, title="imag Jn")
    plt.plot(z_np, res_Jn_np.imag, label="fwd")
    plt.plot(z_np, numdz_dJ_np.imag, label="num. grad.")
    plt.plot(z_np, z_grad_plolt_Jn.imag, label="AD grad.", dashes=[2, 2])
    plt.legend()

    # Yn

    plt.subplot(423, title="real Yn")
    plt.plot(z_np_y, res_Yn_np.real, label="fwd")
    plt.plot(z_np_y, numdz_dY_np.real, label="num. grad.")
    plt.plot(z_np_y, z_grad_plolt_Yn.real, label="AD grad.", dashes=[2, 2])
    plt.legend()

    plt.subplot(424, title="imag Yn")
    plt.plot(z_np_y, res_Yn_np.imag, label="fwd")
    plt.plot(z_np_y, numdz_dY_np.imag, label="num. grad.")
    plt.plot(z_np_y, z_grad_plolt_Yn.imag, label="AD grad.", dashes=[2, 2])
    plt.legend()

    # dJn

    plt.subplot(425, title="real dJn")
    plt.plot(z_np, res_dJn_np.real, label="fwd")
    plt.plot(z_np, numdz_ddJ_np.real, label="num. grad.")
    plt.plot(z_np, z_grad_plolt_dJn.real, label="AD grad.", dashes=[2, 2])
    plt.legend()

    plt.subplot(426, title="imag dJn")
    plt.plot(z_np, res_dJn_np.imag, label="fwd")
    plt.plot(z_np, numdz_ddJ_np.imag, label="num. grad.")
    plt.plot(z_np, z_grad_plolt_dJn.imag, label="AD grad.", dashes=[2, 2])
    plt.legend()

    # dYn

    plt.subplot(427, title="real dYn")
    plt.plot(z_np_y, res_dYn_np.real, label="fwd")
    plt.plot(z_np_y, numdz_ddY_np.real, label="num. grad.")
    plt.plot(z_np_y, z_grad_plolt_dYn.real, label="AD grad.", dashes=[2, 2])
    plt.legend()

    plt.subplot(428, title="imag dYn")
    plt.plot(z_np_y, res_dYn_np.imag, label="fwd")
    plt.plot(z_np_y, numdz_ddY_np.imag, label="num. grad.")
    plt.plot(z_np_y, z_grad_plolt_dYn.imag, label="AD grad.", dashes=[2, 2])
    plt.legend()




    plt.tight_layout()
    plt.show()

    # caution with complex arguments:
    # pytorch uses internally the convention that complex derivatives are wrt the conjugate:
    # derivative = dF/dz*.
    # Therefore, there is a sign difference at the imag part of the output, which needs
    # to be accounted for in the backward pass implementation.
    # see: https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers

    torch.autograd.gradcheck(Jn, (n, z1), eps=0.01)
    torch.autograd.gradcheck(Yn, (n, z2), eps=0.01)
    torch.autograd.gradcheck(dJn, (n, z3), eps=0.01)
    torch.autograd.gradcheck(dYn, (n, z4), eps=0.01)
