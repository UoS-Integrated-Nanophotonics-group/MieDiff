"""
test of autograd optimiser
"""
import warnings

import torch
import numpy as np
import torch.nn.functional as F

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

    # For plotting loss vs iterations
    it = []
    losses = []

    n = torch.tensor([3.0])

    n.requires_grad = True

    # Randomly generate starting z
    temp = np.sort(np.random.uniform(-0.25, 2.25, 100))
    # I had an error without float here
    z = torch.tensor(temp, dtype=float)

    print(z)

    z_real = torch.linspace(-0.25, 2.25, 100, dtype=float)

    print(z_real)

    # Checking random z generation
    #plt.plot(z_real.detach().numpy().squeeze(), -1*(z_real.detach().numpy().squeeze()-1)**2 + 1)
    #plt.plot(z.detach().numpy().squeeze(), -1*(z.detach().numpy().squeeze()-1)**2 + 1)
    #plt.show()

    z_real_np = z_real.detach().numpy().squeeze()
    z.requires_grad = True

    # Target quadratic graph
    target = -1*(z_real-1)**2 + 1

    optimizer = torch.optim.SGD([n, z], lr=0.05, momentum=0.8)

    for i in range(2000):

        optimizer.zero_grad()

        guess = sin_own(z, n)

        loss = F.mse_loss(guess, target)

        loss.backward(retain_graph = True, create_graph = True)

        optimizer.step()

        if i % 10 == 0: # Print every 10 iterations
            it.append(i)
            losses.append(loss.item())
            print(f"Step {i+1}: n = {n.item()}, loss = {loss.item()}")

    # Final result after optimization
    print(f"\nOptimized n = {n.item()}")


    n_np = n.detach().numpy().squeeze()
    z_np = z.detach().numpy().squeeze()
    y_opt = sin_own(z, n).detach().numpy().squeeze()
    y_real = target.detach().numpy().squeeze()

    plt.plot(z_real_np, y_real, label="Real")
    plt.plot(z_np, y_opt, label="Opt.", dashes=[2, 2])
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.plot(it, losses)
    plt.show()
















