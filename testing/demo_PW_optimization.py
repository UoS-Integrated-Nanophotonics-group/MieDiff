# encoding=utf-8
"""
wavelength and mie coefficients vectorization demo

author: P. R. Wiecha, 11/2024
"""
# %%

import torch
import matplotlib.pyplot as plt
import pymiediff as pmd


# - get some reference spectrum as optimization target
k0 = 2 * torch.pi / torch.linspace(400, 800, 50)
r_c0 = torch.tensor(60.0)
r_s0 = torch.tensor(100.0)
n_c0 = torch.tensor(4.0)
n_s0 = torch.tensor(3.0)

res_cs = pmd.coreshell.scs(
    k0=k0,
    r_c=r_c0,
    eps_c=n_c0**2,
    r_s=r_s0,
    eps_s=n_s0**2,
    eps_env=1,
    n_max=5,  # TODO: n_max should be determined automatically inside
)
target = res_cs["q_ext"]


# - initial guess
r_c = torch.tensor(70.0, requires_grad=True)
r_s = torch.tensor(90.0, requires_grad=True)
n_c = torch.tensor(2.5, requires_grad=True)
n_s = torch.tensor(3.5, requires_grad=True)


# - optimization loop
optimizer = torch.optim.Adam([r_c, r_s, n_c, n_s], lr=0.2)

for i in range(301):
    optimizer.zero_grad()

    args = (k0, r_c, n_c**2, r_s, n_s**2)
    qext = pmd.coreshell.scs(*args, n_max=5)["q_ext"]
    loss = torch.nn.functional.mse_loss(target, qext)

    loss.backward(retain_graph=False)
    optimizer.step()

    # - status
    if i % 10 == 0:
        print(i, loss.item())
        plt.figure(figsize=(5, 4))
        plt.subplot(title=f"iteration {i}, loss={loss.item():.3f}")
        plt.plot(target.detach())
        plt.plot(qext.detach())
        plt.show()

print("init :", [f"{d.detach().numpy():.3f}" for d in [r_c0, r_s0, n_c0, n_s0]])
print("final:", [f"{d.detach().numpy():.3f}" for d in [r_c, r_s, n_c, n_s]])
