import torch
import numpy as np
import matplotlib.pyplot as plt

# - dimension 0 will be for angles .unsqueeze(1)
# - dimension 1 will be for mie-order .unsqueeze(0)

def polarPlot(ax, theta, r):
    theta = theta.detach().numpy()
    r = r.detach().numpy()

    r_abs = np.abs(r)
    theta_corrected = np.where(r < 0, theta + np.pi, theta)
    ax.plot(theta_corrected, r_abs)



def pi_tau(N, mu):
    # Ensure N is a scalar integer tensor
    N = int(N)

    # Preallocate the pies tensor with shape (len(mu), N+1)
    pies = torch.zeros(len(mu), N+1, dtype=mu.dtype, device=mu.device)
    taus = torch.zeros(len(mu), N+1, dtype=mu.dtype, device=mu.device)

    # Initialize the first two terms
    pies[:, 0] = 1.0  # π_0 = 1
    taus[:, 0] = mu
    if N > 0:
        pies[:, 1] = 3 * mu  # π_1 = 3 * μ
        taus[:, 1] = 3 * torch.cos(2 * torch.acos(mu))

    # Compute higher-order terms
    for n in range(2, N+1):
        pies[:, n] = ((2 * n + 1) * mu * pies[:, n-1] - (n + 1) * pies[:, n-2]) / n
        taus[:, n] = (n + 1) * mu * pies[:, n] - (n + 2) * pies[:, n-1]
    return pies, taus




N_pt_test = 100
theta = torch.linspace(0.01, 2 * torch.pi, N_pt_test)

pi, tau = pi_tau(torch.tensor(4), torch.cos(theta))


print(pi.shape[1])

fig, ax = plt.subplots(pi.shape[1], 2, subplot_kw={"projection": "polar"}, constrained_layout=True)

for n in range(0, pi.shape[1]):
    polarPlot(ax[n,0], theta, tau[:,n])
    polarPlot(ax[n,1], theta, pi[:,n])

plt.show()

# fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
# polarPlot(ax[0], theta1, temp_t)
# polarPlot(ax[1], theta2, temp_p)
# plt.show()


# fig, ax = plt.subplots(1, 2)
# ax[0].plot(theta1.detach().numpy(), temp_t.detach().numpy())
# ax[1].plot(theta1.detach().numpy(), temp_p.detach().numpy())
# plt.show()