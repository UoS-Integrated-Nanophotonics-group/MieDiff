# encoding=utf-8
"""
wavelength and mie coefficients vectorization demo

author: P. R. Wiecha, 11/2024
"""
# %%

import torch
import matplotlib.pyplot as plt
import pymiediff as pmd
N_pt_test = 100
N_order_test = 8
wl = 400
# - get some reference spectrum as optimization target
k0 = 2 * torch.pi / wl# torch.linspace(400, 800, 50)
theta = torch.linspace(0.01, 2*torch.pi -0.01, N_pt_test)#, dtype=torch.double)
r_c0 = torch.tensor(60.0)
r_s0 = torch.tensor(100.0)
n_c0 = torch.tensor(4.0)
n_s0 = torch.tensor(3.0)

res_angSca = pmd.farfield.angular_scattering(
        k0=k0,
        theta=theta,
        r_c=r_c0,
        eps_c=n_c0**2,
        r_s=r_s0,
        eps_s=n_s0**2,
        eps_env=1.0,
)

target = res_angSca["i_unpol"][0]


# - initial guess
r_c = torch.tensor(70.0, requires_grad=True)
r_s = torch.tensor(90.0, requires_grad=True)
n_c = torch.tensor(3.5, requires_grad=True)
n_s = torch.tensor(2.5, requires_grad=True)
MaxEpoc = 300
lr0 = 0.15
# - optimization loop
optimizer = torch.optim.Adam([r_c, r_s, n_c, n_s], lr=lr0)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                        patience = 100,
#                                                        factor = 0.8)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.66)

losses = []
lrs = []



for i in range(MaxEpoc + 1):
    optimizer.zero_grad()

    kwargs = dict(k0=k0, theta=theta, r_c=r_c, eps_c=n_c**2, r_s=r_s, eps_s=n_s**2)
    i_unp = pmd.farfield.angular_scattering(**kwargs, n_max=N_order_test)["i_unpol"][0]
    loss = torch.nn.functional.mse_loss(target, i_unp)
    losses.append(loss.detach().item())

    loss.backward(retain_graph=False)
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr())

    # - status
    if i % 10 == 0:
        print(i, loss.item())
        fig, ax = plt.subplots(1, constrained_layout=True, figsize=(5, 4), subplot_kw={"projection": "polar"})
        ax.set_title(f"iteration {i}, loss={loss.item():.3f}")
        ax.plot(theta.detach(), target.detach(), label = "target")
        ax.plot(theta.detach(), i_unp.detach(), label = "current iter.")
        ax.legend()
        # plt.savefig('optimiser_plots//iter{:03d}.png'.format(i), dpi = 300)
        plt.show()

print("target:", [f"{d.detach().numpy():.3f}" for d in [r_c0, r_s0, n_c0, n_s0]])
print("final:", [f"{d.detach().numpy():.3f}" for d in [r_c, r_s, n_c, n_s]])

fig, ax1 = plt.subplots(figsize=(5, 4))
ax2 = ax1.twinx()
ax1.set_title("Loss curve.")
ax1.plot(losses, c="blue")
ax2.plot(lrs, c="orange")
ax1.set_xlabel("Iteration Num.")
ax1.set_ylabel("Loss", color="blue")
ax2.set_ylabel("Lr", color="orange")
ax2.set_ylim([0,1])
# plt.savefig('optimiser_plots//lossCurve.png'.format(i))
plt.show()