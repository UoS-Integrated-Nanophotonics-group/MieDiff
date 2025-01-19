import torch
import numpy as np
import matplotlib.pyplot as plt

# - dimension 0 will be for angles .unsqueeze(1)
# - dimension 1 will be for mie-order .unsqueeze(0)

def polarPlot(ax, theta, r,  line = "-"):
    try:
        theta = theta.detach().numpy()
        r = r.detach().numpy()
    except:
        pass

    r_abs = np.abs(r)
    theta_corrected = np.where(r < 0, theta + np.pi, theta)
    ax.plot(theta_corrected, r_abs, linestyle = line)



# def pi_tau(N, mu):
#     # Ensure N is a scalar integer tensor
#     N = int(N)

#     # Preallocate the pies tensor with shape (len(mu), N+1)
#     pies = torch.zeros(len(mu), N+1, dtype=mu.dtype, device=mu.device)
#     taus = torch.zeros(len(mu), N+1, dtype=mu.dtype, device=mu.device)

#     # Initialize the first two terms
#     pies[:, 0] = 1.0  # π_0 = 1
#     taus[:, 0] = mu
#     if N > 0:
#         pies[:, 1] = 3 * mu  # π_1 = 3 * μ
#         taus[:, 1] = 3 * torch.cos(2 * torch.acos(mu))

#     # Compute higher-order terms
#     for n in range(2, N+1):
#         pies[:, n] = ((2 * n + 1) * mu * pies[:, n-1] - (n + 1) * pies[:, n-2]) / n
#         taus[:, n] = (n + 1) * mu * pies[:, n] - (n + 2) * pies[:, n-1]
#     return pies, taus


def pi_tau(N, mu):
    # Ensure N is an integer
    N = int(N)

    # Ensure mu is 1D to avoid shape mismatches
    mu = mu.view(-1)

    # Preallocate tensors for π and τ with the correct shape
    pies = torch.zeros(len(mu), N + 1, dtype=mu.dtype, device=mu.device)
    taus = torch.zeros(len(mu), N + 1, dtype=mu.dtype, device=mu.device)

    # Initialize the first two terms
    pies[:, 0] = 1.0  # π_0 = 1
    taus[:, 0] = mu  # τ_0 = μ
    if N > 0:
        pies[:, 1] = 3 * mu  # π_1 = 3 * μ
        # print(3 * torch.cos(2 * torch.acos(mu)))
        taus[:, 1] = 3 * torch.cos(2 * torch.acos(mu))  # τ_1 = 3cos(2cos⁻¹(μ))

    for n in range(2, N + 1):
        # Compute pies[:, n] out of place
        clone_of_pies = pies.clone()
        pi_n = (
            (2 * n + 1) * mu * clone_of_pies[:, n - 1]
            - (n + 1) * clone_of_pies[:, n - 2]
        ) / n
        pies[:, n] = pi_n

        # Compute taus[:, n] out of place
        clone_of_pies = pies.clone()
        tau_n = (n + 1) * mu * clone_of_pies[:, n] - (n + 2) * clone_of_pies[:, n - 1]
        taus[:, n] = tau_n

    return pies, taus

N_pt_test = 100

def MiePiTau(mu,nmax):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#MiePiTau
  p = np.zeros(int(nmax))
  t = np.zeros(int(nmax))
  # print(p,t)
  p[0] = 1
  p[1] = 3*mu
  t[0] = mu
  t[1] = 3.0*np.cos(2*np.arccos(mu))
  for n in range(2,int(nmax)):
    p[n] = ((2*n+1)*(mu*p[n-1])-(n+1)*p[n-2])/n
    t[n] = (n+1)*mu*p[n]-(n+2)*p[n-1]
  return p, t

theta1_np = np.linspace(0.01, 2*np.pi, N_pt_test)

temp_p = []
temp_t = []

for i in theta1_np:
	p, t = MiePiTau(np.cos(i), 5)
	temp_p.append(p)
	temp_t.append(t)

temp_p = np.array(temp_p)#[:, 4]
temp_t = np.array(temp_t)#[:, 4]


#print(temp_p)
#print(np.conjugate(temp_p))



theta = torch.linspace(0.01, 2 * torch.pi, N_pt_test)

pi, tau = pi_tau(torch.tensor(4), torch.cos(theta))


print(pi.shape[1])

fig, ax = plt.subplots(pi.shape[1], 2, subplot_kw={"projection": "polar"}, constrained_layout=True)

for n in range(0, pi.shape[1]):
    polarPlot(ax[n,0], theta, tau[:,n])
    polarPlot(ax[n,1], theta, pi[:,n])

    polarPlot(ax[n,0], theta, temp_t[:,n], line = "--")
    polarPlot(ax[n,1], theta, temp_p[:,n], line = "--")


plt.show()




# fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
# polarPlot(ax[0], theta1, temp_t)
# polarPlot(ax[1], theta2, temp_p)
# plt.show()


# fig, ax = plt.subplots(1, 2)
# ax[0].plot(theta1.detach().numpy(), temp_t.detach().numpy())
# ax[1].plot(theta1.detach().numpy(), temp_p.detach().numpy())
# plt.show()