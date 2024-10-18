import torch
import numpy as np


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import pymiediff

    N_pt_test = 200
    N_order_test = 1

    n1 = torch.tensor(1)
    n2 = torch.tensor(2)
    n3 = torch.tensor(3)
    n4 = torch.tensor(4)
    n5 = torch.tensor(4)
    n6 = torch.tensor(4)

    wlRes = 1000
    wl = np.linspace(200, 600, wlRes)
    r_core = 12.0
    r_shell = 50.0

    n_env = 1
    n_core = 2 + 0j
    n_shell = 5 + 0j

    mu_env = 1
    mu_core = 1
    mu_shell = 1

    dtype = torch.cfloat  # torch.complex64
    device = torch.device("cpu")

    k = 2 * np.pi / (wl / n_env)
    k = torch.tensor(k, dtype=dtype)

    m1 = n_core / n_env
    m2 = n_shell / n_env

    r_c = torch.tensor(r_core, requires_grad=True, dtype=dtype)
    r_s = torch.tensor(r_shell, requires_grad=True, dtype=dtype)

    x = k * r_c
    y = k * r_s

    m1 = torch.tensor(m1, requires_grad=True, dtype=dtype)
    m2 = torch.tensor(m2, requires_grad=True, dtype=dtype)

    a1 = pymiediff.an(x, y, n1, m1, m2)
    a2 = pymiediff.an(x, y, n2, m1, m2)
    a3 = pymiediff.an(x, y, n3, m1, m2)
    a4 = pymiediff.an(x, y, n4, m1, m2)

    b1 = pymiediff.bn(x, y, n1, m1, m2)
    b2 = pymiediff.bn(x, y, n2, m1, m2)
    b3 = pymiediff.bn(x, y, n3, m1, m2)
    b4 = pymiediff.bn(x, y, n4, m1, m2)

    def cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4):
        return ((2*torch.pi/k**2) *
                ((2*n1 + 1)*(a1.real**2 + a1.imag**2 + b1.real**2 + b1.imag**2) +
                 (2*n2 + 1)*(a2.real**2 + a2.imag**2 + b2.real**2 + b2.imag**2) +
                 (2*n3 + 1)*(a3.real**2 + a3.imag**2 + b3.real**2 + b3.imag**2) +
                 (2*n4 + 1)*(a4.real**2 + a4.imag**2 + b4.real**2 + b4.imag**2)))

    Csca = cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4)
    Csca_np = Csca.detach().numpy()

    checkAutograd = False
    checkForward = False
    checkGrad = False

    if checkAutograd:
        check = torch.autograd.gradcheck(cross_sca, [k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4], eps=0.01)
        print("autograd.gradcheck positive?", check)

    if checkGrad:
        r_c_grad = torch.autograd.grad(outputs=Csca,
                                       inputs=[r_c, r_s, m1, m2],
                                       grad_outputs=torch.ones_like(Csca))
        print(r_c_grad)

    if checkForward:
        import PyMieScatt as ps

        CscaReal = []
        for wavelengh in wl:
            scatter = ps.MieQCoreShell(mCore=n_core,
                                       mShell=n_shell,
                                       wavelength=wavelengh,
                                       dCore=2 * r_core,
                                       dShell=2 * r_shell,
                                       nMedium=n_env,
                                       asCrossSection=False,
                                       asDict=True)["Qsca"]

            CscaReal.append(scatter)

        plt.plot(Csca_np.real/(np.pi*(r_shell)**2))
        plt.plot(CscaReal, ls="--")
        plt.show()
