import pymiediff as pmd
import torch
import matplotlib.pyplot as plt

n_max = 10

def NumCenterDiff(Funct, n, z, eps=0.0001 + 0.0001j):
    z = z.conj()
    fm_0, fm_1, = Funct(n, z - eps)
    fp_0, fp_1 = Funct(n, z + eps)
    dz_0 = (fp_0 - fm_0) / (2 * eps)
    dz_1 = (fp_1 - fm_1) / (2 * eps)
    return dz_0, dz_1

N_pt_test = 50

fig, ax = plt.subplots(2, figsize=(9, 9), dpi=100, constrained_layout=True)

funct = pmd.angular.pi_tau

N_pt_test = 100

for n in range(10, 20):


    mu = torch.cos(torch.linspace(0.01, 2 * torch.pi, N_pt_test))
    mu.requires_grad = True

    result = funct(n, mu)
    num_grad = NumCenterDiff(funct, n, mu)


    grad = torch.autograd.grad(
        outputs=result, inputs=[mu], grad_outputs=(torch.ones_like(result[0]),torch.ones_like(result[0]))
    )

    # result_np = result.detach().numpy().squeeze()
    mu_np_0 = mu[0].detach().numpy().squeeze()
    mu_np_1 = mu[1].detach().numpy().squeeze()
    num_grad_np_0 = num_grad[0].detach().numpy().squeeze()
    num_grad_np_1 = num_grad[1].detach().numpy().squeeze()
    grad_np_0 = grad[0][0].detach().numpy().squeeze()
    grad_np_1 = grad[0][1].detach().numpy().squeeze()


    #ax[0].plot(num_grad_np_0)
    ax[0].plot(grad_np_0, linestyle = ":", label = "ag", linewidth = 2.0)
    ax[0].legend()

    #ax[1].plot(num_grad_np_1)
    ax[1].plot(grad_np_1, linestyle = ":", label = "ag", linewidth = 2.0)
    ax[1].legend()

plt.show()








