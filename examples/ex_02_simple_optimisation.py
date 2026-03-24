# encoding: utf-8
"""
Minimal optimization
====================

Pedagogical example of gradient-based optimization with pymiediff:
optimize the radius of a homogeneous sphere to maximize scattering at
one wavelength.

author: P. Wiecha, 03/2026
"""
# %%
# imports
# -------
import torch
import pymiediff as pmd

# %%
# setup
# -----
wl0 = torch.tensor([700.0])  # nm
k0 = 2 * torch.pi / wl0

n_p = 3.5
n_env = 1.0


# %%
# optimization
# ------------

# initial guess
r_opt = torch.tensor([60.0], requires_grad=True)
optimizer = torch.optim.Adam([r_opt], lr=0.5)

for i in range(100):
    optimizer.zero_grad()
    particle = pmd.Particle(
        mat_env=n_env,
        r_layers=r_opt,
        mat_layers=[n_p],
    )
    q_sca = particle.get_cross_sections(k0)["q_sca"]

    loss = -q_sca  # *maximize* scattering: minus sign
    loss.backward()
    optimizer.step()

    print(f"iter {i:02d}: r = {r_opt} nm, Q_sca = {q_sca}")


# %%
# result
# ------
print(f"\nOptimized radius: {r_opt.item():.2f} nm")
print(f"Scattering at {wl0.item():.1f} nm: Q_sca = {q_sca.item():.4f}")
