import torch
import numpy as np
import matplotlib.pyplot as plt

def pi_n(N, theta):
    N = torch.as_tensor(N, dtype=torch.int)
    if N == 0:
        return torch.zeros_like(theta, dtype=theta.dtype, device=theta.device)
    elif N == 1:
        return torch.ones_like(theta, dtype=theta.dtype, device=theta.device)
    pi_nm2 = torch.zeros_like(theta, dtype=theta.dtype, device=theta.device)
    pi_nm1 = torch.ones_like(theta, dtype=theta.dtype, device=theta.device)

    for n in range(2, N + 1):
        pi_n = ((2 * n - 1) / (n - 1)) * torch.cos(theta) * pi_nm1 - (
            n / (n - 1)
        ) * pi_nm2
        pi_nm2 = pi_nm1
        pi_nm1 = pi_n
    return pi_n


def tau_n(N, theta):
    N = torch.as_tensor(N, dtype=torch.int)
    if N == 0:
        return torch.zeros_like(theta, dtype=theta.dtype, device=theta.device)
    for n in range(1, N + 1):
        tau_n = n * torch.cos(theta) * pi_n(n, theta) - (n - 1) * pi_n(n - 1, theta)
    return tau_n

N_pt_test = 50

n = torch.tensor(5)
#pi
theta1 = torch.linspace(0.01, 2*torch.pi, N_pt_test)
# theta1.requires_grad = True
#tau
theta2 = torch.linspace(0.01, 2*torch.pi, N_pt_test)
# theta2.requires_grad = True

fig, ax = plt.subplots(2, subplot_kw={'projection': 'polar'})

ax[0].plot(theta1.detach().numpy(), pi_n(n, theta1).detach().numpy())
ax[1].plot(theta2.detach().numpy(), tau_n(n, theta2).detach().numpy())

plt.show()



