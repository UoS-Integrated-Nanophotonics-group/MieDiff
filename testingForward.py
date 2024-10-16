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
    r_shell = 50.0 # + r_core # r_core + 100.0

    n_env = 1
    n_core = 2
    n_shell = 5 # 0.1 + 0.7j

    mu_env = 1
    mu_core = 1
    mu_shell = 1

    dtype = torch.complex64
    device = torch.device("cpu")

    k = 2 * np.pi / (wl / n_env)

    # print(k)

    m1 = n_core / n_env
    m2 = n_shell / n_env
    x = k * r_core
    y = k * r_shell

    x = torch.tensor(x, requires_grad=True, dtype=dtype)
    y = torch.tensor(y, requires_grad=True, dtype=dtype)
    m1 = torch.tensor(m1, requires_grad=True, dtype=dtype)
    m2 = torch.tensor(m2, requires_grad=True, dtype=dtype)

    k = torch.tensor(k, dtype=dtype)

    # print(m2)
    # print(m2.dtype)

    # print(x)

    # a1 = psi(n, x)
    a1 = pymiediff.an(x, y, n1, m1, m2)
    a2 = pymiediff.an(x, y, n2, m1, m2)
    a3 = pymiediff.an(x, y, n3, m1, m2)
    a4 = pymiediff.an(x, y, n4, m1, m2)
    a5 = pymiediff.an(x, y, n5, m1, m2)
    a6 = pymiediff.an(x, y, n6, m1, m2)

    b1 = pymiediff.bn(x, y, n1, m1, m2)
    b2 = pymiediff.bn(x, y, n2, m1, m2)
    b3 = pymiediff.bn(x, y, n3, m1, m2)
    b4 = pymiediff.bn(x, y, n4, m1, m2)
    b5 = pymiediff.bn(x, y, n5, m1, m2)
    b6 = pymiediff.bn(x, y, n6, m1, m2)

    Csca = (2*torch.pi/k**2) * ((2*n1 + 1)*(a1.real**2 + a1.imag**2 + b1.real**2 + b1.imag**2) +
                                (2*n2 + 1)*(a2.real**2 + a2.imag**2 + b2.real**2 + b2.imag**2) +
                                (2*n3 + 1)*(a3.real**2 + a3.imag**2 + b3.real**2 + b3.imag**2) +
                                (2*n4 + 1)*(a4.real**2 + a4.imag**2 + b4.real**2 + b4.imag**2) +
                                (2*n5 + 1)*(a5.real**2 + a5.imag**2 + b5.real**2 + b5.imag**2) +
                                (2*n6 + 1)*(a6.real**2 + a6.imag**2 + b6.real**2 + b6.imag**2))

    # print(Csca)

    Csca = Csca.detach().numpy()
    import PyMieScatt as ps

    CscaReal = []
    for wavelengh in wl:
        scatter = ps.MieQCoreShell( mCore=n_core,
                                    mShell=n_shell,
                                    wavelength=wavelengh,
                                    dCore=2 * r_core,
                                    dShell=2 * r_shell,
                                    nMedium=n_env,
                                    asCrossSection=False,
                                    asDict=True)["Qsca"]
        # print(wavelengh)
        # print(scatter)
        CscaReal.append(scatter)


    # CscaReal = Q_dict["Qsca"]
    # print(Q_dict["Qext"], Q_dict["Qsca"], Q_dict["Qabs"])

    plt.plot(Csca.real/(np.pi*(r_shell)**2))
    plt.plot(CscaReal, ls="--")
    plt.show()


