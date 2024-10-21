import numpy as np
import torch


def PlotScatteringCrossSection(ax, radi, m,  waveLengths, scattering, max_dis = 2, multipoles = None, norm = 1, sizeParameter = False, prefix = "nm"):
    keys = [["ED",(0, (3, 5, 1, 5, 1, 5)),"firebrick" ,"MD",(0, (3, 1, 1, 1, 1, 1)),"royalblue" ],
            ["EQ",(0, (3, 5, 1, 5)),"maroon" ,"MQ",(0, (3, 1, 1, 1)), "navy"],
            ["EO",(0, (5, 1)),"indianred" ,"MO",(0, (5, 5)),"cornflowerblue"],
            ["E16",(0, (1, 1)),"lightcoral" ,"M16" ,(0, (1, 5)),"slateblue"]]

    if sizeParameter:
        var = "Size Parameter $ x=k a=\\frac{2 \pi n a}{\lambda}$"
    else:
        var = "$\lambda$"

    if multipoles is not None:
        n_num = len(multipoles)
        for n in range(1,n_num+1):
            if (n <= max_dis) and (n <= 4):
                ax.plot(waveLengths, multipoles[n-1][0]/norm, label = keys[n-1][0], linestyle = keys[n-1][1], color=keys[n-1][2], linewidth=0.8)
                ax.plot(waveLengths, multipoles[n-1][1]/norm, label = keys[n-1][3], linestyle = keys[n-1][4], color=keys[n-1][5], linewidth=0.8)
            elif (n <= max_dis):
                ax.plot(waveLengths, multipoles[n-1][0]/norm, color="red", linewidth=0.8)
                ax.plot(waveLengths, multipoles[n-1][1]/norm, color="blue", linewidth=0.8)

    ax.plot(waveLengths, scattering/norm, c = "black", label = "Total",linewidth=0.8)
    ax.set_xlabel(var)
    if norm != 1:
        ax.set_ylabel("$\sigma_{s}/r_{shell}^2$ ")
    else:
        ax.set_ylabel("$\sigma_{s}$ ($m^{2}$)")
    ax.set_title("Scattering cross section of shell sphere of radi, $r_i = {}$ nm, and corresponiding relative reflection index $m_i = {}$".format(str(radi)[1:-1], str(m)[1:-1]), fontsize=10)
    #ax.set_title("Scattering cross section of sphere of raduis, a = " + str(a) + prefix + " and $m=n_{1}/n=$" + str(m) + "\n calculated from Mie-Theroy displaying up to n = " + str(max_dis))
    #ax.set_title("Scattering cross section of a sphere showing the different scattering regimes")
    ax.legend()


def MakeMultipoles(As, Bs, k):
    multipoles = []
    for n, (a, b) in enumerate(zip(As,Bs)):
        multipoles.append([(2*np.pi/k.detach().numpy()**2) * (2*(n + 1) + 1)* (a.abs()**2).detach().numpy(),
                           (2*np.pi/k.detach().numpy()**2) * (2*(n + 1) + 1)* (b.abs()**2).detach().numpy() ])
    return multipoles


# numerical center diff. for testing:
def NumCenterDiff(Funct, n, z, eps=0.0001 + 0.0001j):
    z = z.conj()
    fm = Funct(n, z - eps)
    fp = Funct(n, z + eps)
    dz = (fp - fm) / (2 * eps)
    return dz


def GradCheckerPlot(ax, z, fwd, grad, num_grad, name,  re = False, im = False):
    if re:
        ax.set_title("Real {}".format(name))
        ax.plot(z, fwd.real, label="Forward")
        ax.plot(z, num_grad.real, label="Num. grad.")
        ax.plot(z, grad.real, label="AD grad.", dashes=[2, 2])
        ax.set_xlabel("z")
        ax.legend()
    if im:
        ax.set_title("Imag. {}".format(name))
        ax.plot(z, fwd.imag, label="Forward")
        ax.plot(z, num_grad.imag, label="Num. grad.")
        ax.plot(z, grad.imag, label="AD grad.", dashes=[2, 2])
        ax.set_xlabel("z")
        ax.legend()


def FunctGradChecker(z, funct, inputs, ax = None, real = False, imag = False):
    result = funct(*inputs)
    num_grad = NumCenterDiff(funct, *inputs)
    grad = torch.autograd.grad(outputs=result,
                               inputs=[z],
                               grad_outputs=torch.ones_like(result)
                               )

    result_np = result.detach().numpy().squeeze()
    z_np = z.detach().numpy().squeeze()
    num_grad_np = num_grad.detach().numpy().squeeze()
    grad_np = grad[0].detach().numpy().squeeze()

    if ax is not None:
        GradCheckerPlot(ax, z_np, result_np, grad_np, num_grad_np, funct.__name__, re = real, im = imag)




    return z_np, num_grad_np, grad_np











