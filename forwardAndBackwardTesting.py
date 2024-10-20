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
                ((2*n1 + 1)*(a1.abs()**2 + b1.abs()**2) +
                 (2*n2 + 1)*(a2.abs()**2 + b2.abs()**2) +
                 (2*n3 + 1)*(a3.abs()**2 + b3.abs()**2) +
                 (2*n4 + 1)*(a4.abs()**2 + b4.abs()**2)))

    Csca = cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4)
    Csca_np = Csca.detach().numpy()

    checkAutograd = False
    checkForward, compareForward = True, False
    checkGrad = False

    if checkAutograd:
        check = torch.autograd.gradcheck(cross_sca, [k,
                                                     n1, a1, b1,
                                                     n2, a2, b2,
                                                     n3, a3, b3,
                                                     n4, a4, b4],
                                         eps=0.01)
        print("autograd.gradcheck positive?", check)

    if checkGrad:
        r_c_grad = torch.autograd.grad(outputs=Csca,
                                       inputs=[r_c, r_s, m1, m2],
                                       grad_outputs=torch.ones_like(Csca))
        print(r_c_grad)

    if checkForward:
        if compareForward:
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
            plt.plot(CscaReal, ls="--")

        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)


        multipoles = pymiediff.helper.MakeMultipoles([a1, a2, a3, a4],
                                                     [b1, b2, b3, b4],
                                                     k)

        pymiediff.helper.PlotScatteringCrossSection(ax,
                                                    (r_core,
                                                     r_shell),
                                                    (m1.detach().numpy().item(),
                                                     m2.detach().numpy().item()),
                                                    wl, Csca_np.real,
                                                    max_dis = 3,
                                                    multipoles = multipoles,
                                                    norm=np.pi*(r_shell)**2)

        plt.show()
