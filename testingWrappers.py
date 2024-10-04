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

        grad_input = torch.from_numpy(grad_input) \
            * torch.from_numpy(grad_output)

        return grad_input, None  # No gradient w.r.t. n


# Testing gradcheck
input = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
n = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float64, requires_grad=False)

thing = torch.autograd.gradcheck(torch_jn.apply, (input, n))
print(thing)
