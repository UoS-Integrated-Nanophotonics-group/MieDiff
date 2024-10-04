from scipy.special import spherical_jn, spherical_yn

from scipy import special

import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import numpy as np


class torch_jn(Function):
    @staticmethod
    def forward(ctx, input, n):
        input_np = input.detach().numpy()
        result = torch.from_numpy(spherical_jn(n, input_np))
        ctx.save_for_backward(input)
        ctx.n = n  # Save n (non-learnable) in ctx

        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output = grad_output.detach().numpy()

        # Derivative of spherical_jn w.r.t. input
        input_np = input.detach().numpy()
        grad_input = spherical_jn(ctx.n, input_np, derivative=True)

        # In automatic differentiation, the `grad_output` tensor
        # represents the gradient of the loss with respect to the output of the
        # function. To obtain the gradient of the loss with respect to the
        # input, you need to multiply this `grad_output` by the local
        # derivative of your function with respect to the input, which is
        # represented by `grad_input`.
        grad_input = torch.from_numpy(grad_input) \
            * torch.from_numpy(grad_output)

        return grad_input, None  # No gradient w.r.t. n


# Testing gradcheck
input_res = 10

np_input = np.linspace(3, 5, input_res)

np_input = np.reshape(np_input, newshape=(input_res, 1))

input = torch.tensor(np_input, dtype=torch.float64, requires_grad=True)
n = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float64, requires_grad=False)

thing = torch.autograd.gradcheck(torch_jn.apply, (input, n))
print(thing)
