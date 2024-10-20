import torch
import numpy as np


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pymiediff

    N_pt_test = 50
    # n
    n = torch.tensor(5.0)
    # Jn
    z1 = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z1.requires_grad = True
    # dJn
    z2 = torch.linspace(1, 20, N_pt_test) + 1j * torch.linspace(0.5, 3, N_pt_test)
    z2.requires_grad = True

    # Yn
    z3 = torch.linspace(0.5, 2, N_pt_test) + 1j * torch.linspace(0.5, 2, N_pt_test)
    z3.requires_grad = True
    # dYn
    z4 = torch.linspace(0.5, 2, N_pt_test) + 1j * torch.linspace(0.5, 2, N_pt_test)
    z4.requires_grad = True


    fig, ax = plt.subplots(4, 2, figsize=(16, 9), dpi=100, constrained_layout=True)

    pymiediff.helper.FunctGradChecker(z1, pymiediff.special.Jn, (n, z1), ax = ax[0,0], real = True)
    pymiediff.helper.FunctGradChecker(z1, pymiediff.special.Jn, (n, z1), ax = ax[0,1], imag = True)
    pymiediff.helper.FunctGradChecker(z2, pymiediff.special.dJn, (n, z2), ax = ax[1,0], real = True)
    pymiediff.helper.FunctGradChecker(z2, pymiediff.special.dJn, (n, z2), ax = ax[1,1], imag = True)
    pymiediff.helper.FunctGradChecker(z3, pymiediff.special.Yn, (n, z3), ax = ax[2,0], real = True)
    pymiediff.helper.FunctGradChecker(z3, pymiediff.special.Yn, (n, z3), ax = ax[2,1], imag = True)
    pymiediff.helper.FunctGradChecker(z4, pymiediff.special.dYn, (n, z4), ax = ax[3,0], real = True)
    pymiediff.helper.FunctGradChecker(z4, pymiediff.special.dYn, (n, z4), ax = ax[3,1], imag = True)

    plt.show()


