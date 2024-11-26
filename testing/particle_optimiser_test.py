# %%
import torch
import numpy as np
import time
import torch.nn.functional as F

# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import pymiediff

    it = []
    losses = []

    N_pt_test = 50
    N_order_test = 1

    n1 = torch.tensor(1)
    n2 = torch.tensor(2)
    n3 = torch.tensor(3)
    n4 = torch.tensor(4)
    n5 = torch.tensor(4)
    n6 = torch.tensor(4)

    wlRes = 100
    wl = np.linspace(400, 800, wlRes)
    r_core = 80.0
    r_shell = r_core + 10.0

    n_env = 1
    n_core = 3 + 0j
    n_shell = 2.0

    dtype = torch.complex128  # torch.complex64
    device = torch.device("cpu")

    k = 2 * np.pi / (wl / n_env)
    k = torch.tensor(k, dtype=dtype)

    m1 = n_core / n_env
    m2 = n_shell / n_env

    r_c = torch.tensor(r_core, requires_grad=True, dtype=torch.double)
    r_s = torch.tensor(r_shell, requires_grad=True, dtype=torch.double)

    x = k * r_c
    y = k * r_s

    m1 = torch.tensor(m1, requires_grad=False, dtype=dtype)
    m2 = torch.tensor(m2, requires_grad=False, dtype=dtype)
    t0 = time.time()
    a1 = pymiediff.coreshell.an(x, y, n1, m1, m2)
    a2 = pymiediff.coreshell.an(x, y, n2, m1, m2)
    a3 = pymiediff.coreshell.an(x, y, n3, m1, m2)
    a4 = pymiediff.coreshell.an(x, y, n4, m1, m2)

    b1 = pymiediff.coreshell.bn(x, y, n1, m1, m2)
    b2 = pymiediff.coreshell.bn(x, y, n2, m1, m2)
    b3 = pymiediff.coreshell.bn(x, y, n3, m1, m2)
    b4 = pymiediff.coreshell.bn(x, y, n4, m1, m2)

    Csca = pymiediff.coreshell.cross_sca(
        k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4
    )

    import PyMieScatt as ps

    y_real = []
    for wavelengh in wl:
        scatter = ps.MieQCoreShell(
            mCore=n_core,
            mShell=n_shell,
            wavelength=wavelengh,
            dCore=2.1 * r_core,
            dShell=2.1 * r_shell,
            nMedium=n_env,
            asCrossSection=False,
            asDict=True,
        )["Qsca"]

        y_real.append(scatter)

    CscaTarget = torch.tensor(y_real)

    y_start = (
        pymiediff.coreshell.cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4)
        .detach()
        .numpy()
    )

    # plt.plot(wl, y_real, label="target")
    # plt.plot(
    #    wl, y_start / (np.pi * (r_shell) ** 2), label="pymiediff-start.", dashes=[2, 2]
    # )
    plt.legend()
    # %%
    optimizer = torch.optim.SGD([r_c, r_s, m1, m2], lr=0.01)
    optimizer = torch.optim.Adam([r_c, r_s], lr=0.4)
    norm = 1.0

    for i in range(200):

        if r_s < r_c:
            size_constraint = (r_s - r_c) * 10
            r_s_constrained = r_c.item()
        else:
            r_s_constrained = r_s
            size_constraint = 0

        x = k * r_c
        y = k * r_s_constrained

        a1 = pymiediff.coreshell.an(x, y, n1, m1, m2)
        a2 = pymiediff.coreshell.an(x, y, n2, m1, m2)
        a3 = pymiediff.coreshell.an(x, y, n3, m1, m2)
        a4 = pymiediff.coreshell.an(x, y, n4, m1, m2)

        b1 = pymiediff.coreshell.bn(x, y, n1, m1, m2)
        b2 = pymiediff.coreshell.bn(x, y, n2, m1, m2)
        b3 = pymiediff.coreshell.bn(x, y, n3, m1, m2)
        b4 = pymiediff.coreshell.bn(x, y, n4, m1, m2)

        optimizer.zero_grad()

        guess_scs = pymiediff.coreshell.cross_sca(
            k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4
        )
        guess_qsca = guess_scs / (torch.pi * (r_shell) ** 2)
        loss = torch.mean(
            torch.abs((guess_qsca.real / norm - CscaTarget.real / norm)) ** 2
        )
        loss = loss - size_constraint

        loss.backward()
        # loss.backward(retain_graph=True, create_graph=True)

        optimizer.step()

        it.append(i)
        losses.append(loss.item())
        if i % 10 == 0:  # Print every 10 iterations
            print(
                "Step {}: r_c, r_s, m1, m2 = {}, {}, {}, {} loss = {}, size_constraint={}".format(
                    i + 1,
                    r_c.item(),
                    r_s.item(),
                    m1.item(),
                    m2.item(),
                    loss.item(),
                    size_constraint,
                )
            )

    y_opt = (
        pymiediff.coreshell.cross_sca(k, n1, a1, b1, n2, a2, b2, n3, a3, b3, n4, a4, b4)
        .detach()
        .numpy()
    )

    plt.plot(
        wl,
        y_real,
        label="target  r_c, r_s, m1, m2 = {}, {}, {}, {}".format(
            r_core, r_shell, n_core / n_env, n_shell / n_env
        ),
    )
    plt.plot(
        wl,
        y_start / (np.pi * (r_shell) ** 2),
        label="pymiediff-start. r_c, r_s, m1, m2 = {}, {}, {}, {}".format(
            1.05 * r_core, 1.05 * r_shell, n_core / n_env, n_shell / n_env
        ),
        dashes=[2, 2],
    )
    plt.plot(
        wl,
        y_opt / (np.pi * (r_shell) ** 2),
        label="Opt. r_c, r_s, m1, m2 = {}, {}, {}, {}".format(
            r_c.item(), r_s.item(), m1.item(), m2.item()
        ),
        dashes=[2, 2],
    )
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.plot(it, losses)
    plt.show()
