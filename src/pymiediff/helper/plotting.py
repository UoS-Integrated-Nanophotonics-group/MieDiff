# -*- coding: utf-8 -*-
"""
plotting stuff
"""

plot_styles = [
    [
        "ED",
        (0, (3, 5, 1, 5, 1, 5)),
        "firebrick",
        "MD",
        (0, (3, 1, 1, 1, 1, 1)),
        "royalblue",
    ],
    ["EQ", (0, (3, 5, 1, 5)), "maroon", "MQ", (0, (3, 1, 1, 1)), "navy"],
    ["EO", (0, (5, 1)), "indianred", "MO", (0, (5, 5)), "cornflowerblue"],
    ["E16", (0, (1, 1)), "lightcoral", "M16", (0, (1, 5)), "slateblue"],
]


def _get_axis_existing_or_new_axes():
    import matplotlib.pyplot as plt

    if len(plt.get_fignums()) == 0:
        show = True
        ax = plt.subplot()
    else:
        show = False
        ax = plt.gca()
    return ax, show



def plot_cross_section(
    ax,
    radi,
    ns,
    waveLengths,
    scattering,
    names=["total"],
    max_dis=2,
    multipoles=None,
    norm=1,
    prefix="nm",
    title=None,
):
    from pymiediff.helper.helper import detach_tensor

    radi = detach_tensor(radi, item=True)
    ns = detach_tensor(ns, item=True)
    waveLengths = waveLengths.detach().numpy()
    scattering = detach_tensor(scattering)

    var = "$\lambda$ ({})".format(prefix)
    if multipoles is not None:
        n_num = multipoles.shape[-1]
        for n in range(1, n_num + 1):
            if (n <= max_dis) and (n <= 4):
                ax.plot(
                    waveLengths,
                    multipoles[0, :, n - 1] / norm,
                    label=plot_styles[n - 1][0],
                    linestyle=plot_styles[n - 1][1],
                    color=plot_styles[n - 1][2],
                    linewidth=0.8,
                )
                ax.plot(
                    waveLengths,
                    multipoles[1, :, n - 1] / norm,
                    label=plot_styles[n - 1][3],
                    linestyle=plot_styles[n - 1][4],
                    color=plot_styles[n - 1][5],
                    linewidth=0.8,
                )
            elif n <= max_dis:
                ax.plot(
                    waveLengths,
                    multipoles[0, :, n - 1] / norm,
                    color="red",
                    linewidth=0.8,
                )
                ax.plot(
                    waveLengths,
                    multipoles[1, :, n - 1] / norm,
                    color="blue",
                    linewidth=0.8,
                )
    if isinstance(scattering, tuple):
        for i, name in zip(scattering, names):
            ax.plot(waveLengths, i / norm, label=name, linewidth=2)
    else:
        ax.plot(waveLengths, scattering / norm, label=names, linewidth=2)

    ax.set_xlabel(var)
    if norm != 1:
        ax.set_ylabel("$\sigma_{s}/r_{shell}^2$ ")
    else:
        ax.set_ylabel("$\sigma_{s}$ ($m^{2}$)")
    if title is None:
        ax.set_title(
            "Scattering cross section of shell sphere of radi, $r_i = {}$ nm, \n and corresponiding reflection index $n_i = {}$".format(
                str(radi)[1:-1],
                str([round(elem.real, 4) + round(elem.imag, 4) * 1j for elem in ns])[
                    1:-1
                ],
            ),
            fontsize=10,
        )
    else:
        ax.set_title(
            title,
            fontsize=10,
        )
    ax.legend()


def plot_grad_checker(ax1, ax2, z, fwd, grad, num_grad, name, imag=True, check=None):
    ax1.set_title("Real {}. Passed grad check: {}".format(name, check))
    ax1.plot(z, fwd.real, label="Forward")
    ax1.plot(z, num_grad.real, label="Num. grad.")
    ax1.plot(z, grad.real, label="AD grad.", dashes=[2, 2])
    ax1.set_xlabel("z")
    ax1.legend()

    if not imag:
        ax2.set_axis_off()
        return

    ax2.set_title("Imag. {}. Passed grad check: {}".format(name, check))
    ax2.plot(z, fwd.imag, label="Forward")
    ax2.plot(z, num_grad.imag, label="Num. grad.")
    ax2.plot(z, grad.imag, label="AD grad.", dashes=[2, 2])
    ax2.set_xlabel("z")
    ax2.legend()


def plot_angular(
    ax, radi, ns, wavelength, angles, scattering, prefix="nm", names=None, title=None
):
    from pymiediff.helper.helper import detach_tensor

    radi = detach_tensor(radi, item=True)
    ns = detach_tensor(ns, item=True)
    wavelength = wavelength.detach().numpy()
    angles = angles.detach().numpy()
    scattering = detach_tensor(scattering)

    if isinstance(scattering, tuple):
        for i, name in zip(scattering, names):
            ax.plot(angles, i, label=name, linewidth=2)
    else:
        ax.plot(angles, scattering, label=names, linewidth=2)

    if title == "Auto":
        ax.set_title(
            "Angular response of shell sphere of $r_i = {}$ nm, with $n_i = {}$ at $\lambda = {} {}$".format(
                str(radi)[1:-1],
                str([round(elem.real, 4) + round(elem.imag, 4) * 1j for elem in ns])[
                    1:-1
                ],
                wavelength,
                prefix,
            ),
            fontsize=10,
        )
    elif title is not None:
        ax.set_title(
            title,
            fontsize=10,
        )
    ax.legend()
