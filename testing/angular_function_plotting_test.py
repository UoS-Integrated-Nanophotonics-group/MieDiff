import torch
import numpy as np
import matplotlib.pyplot as plt
import pymiediff as pmd

# Correction to allow polar plot to pass through origin when r is negative
def polar_plot(ax, theta, r, line="-", title = None):
    try:
        theta = theta.detach().numpy()
        r = r.detach().numpy()
    except:
        pass

    r_abs = np.abs(r)
    theta_corrected = np.where(r < 0, theta + np.pi, theta)
    ax.plot(theta_corrected, r_abs, linestyle=line)
    if title is not None:
        ax.set_title(title)


N_pt_test = 100
theta = torch.linspace(0.01, 2 * torch.pi, N_pt_test)

pi, tau = pmd.special.pi_tau(torch.tensor(4), torch.cos(theta))

fig, ax = plt.subplots(
    pi.shape[0],
    2,
    subplot_kw={"projection": "polar"},
    constrained_layout=True,
    figsize=(5, 10),
)

for n in range(0, pi.shape[0]):
    polar_plot(ax[n, 0], theta, tau[n, :], title="$τ_{}$".format(n))
    ax[n,0].set_xticks([0, np.pi/4, np.pi])
    polar_plot(ax[n, 1], theta, pi[n, :], title="$π_{}$".format(n))
    ax[n,1].set_xticks([0, np.pi/4, np.pi])
plt.show()

#%%
import torch
import pymiediff as pmd
import matplotlib.pyplot as plt

# - setup the particle
mat_core = pmd.materials.MatDatabase("Si")
mat_shell = pmd.materials.MatDatabase("Ge")

p = pmd.Particle(
    r_layers=[10.0, 80.0],  # nm
    mat_layers=[mat_core, mat_shell],
)

# - calculate efficiencies / cross section spectra
wl = torch.linspace(500, 1000, 50)
cs = p.get_cross_sections(k0=2 * torch.pi / wl)

plt.plot(cs["wavelength"], cs["q_ext"], label="$Q_{ext}$")


# - autodiff quick test
wl = torch.as_tensor(500.0)
wl.requires_grad = True
cs = p.get_cross_sections(k0=2 * torch.pi / wl)

cs["q_sca"].backward()
dQdWl = wl.grad