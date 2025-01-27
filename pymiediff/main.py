# -*- coding: utf-8 -*-
"""
main routines of pymiediff
"""
import warnings
import numpy as np
import torch
from typing import Union

# from . import special
from pymiediff import special  # use absolute package internal imports!
from pymiediff import coreshell
from pymiediff import angular
from pymiediff import farfield

# define here functions / classes that should be provided by
# the `main` module of the package


def seedComb(
    r_c_min, r_c_max, r_s_min, r_s_max, n_c_min, n_c_max, n_s_min, n_s_max, NumComb=100
):
    # Generate random values for r_c and r_s
    r_c0 = np.random.uniform(r_c_min, r_c_max, NumComb)
    r_s0 = np.random.uniform(r_s_min, r_s_max, NumComb)

    # Generate random values for n_c and n_s (real and imaginary parts separately)
    n_c_real = np.random.uniform(n_c_min.real, n_c_max.real, NumComb)
    n_c_imag = np.random.uniform(n_c_min.imag, n_c_max.imag, NumComb)
    n_c0 = n_c_real + 1j * n_c_imag

    n_s_real = np.random.uniform(n_s_min.real, n_s_max.real, NumComb)
    n_s_imag = np.random.uniform(n_s_min.imag, n_s_max.imag, NumComb)
    n_s0 = n_s_real + 1j * n_s_imag

    return r_c0, r_s0, n_c0, n_s0


class particle:
    def __init__(
        self,
        r_c: Union[list, np.ndarray, torch.Tensor],
        r_s: Union[list, np.ndarray, torch.Tensor],
        n_c: Union[list, np.ndarray, torch.Tensor],
        n_s: Union[list, np.ndarray, torch.Tensor],
    ) -> None:

        assert (
            len(r_c) == len(r_s) and len(n_c) == len(r_c) and len(r_c) == len(r_c)
        ), "parameter inputs must be same size"

        self.core_radius = r_c  # nm
        self.shell_radius = r_s  # nm
        self.core_refractiveIndex = n_c
        self.shell_refractiveIndex = n_s

    def pre_optimise(self, target, parameter_range):
        return

    def optimise(
        self,
        k0,
        target,
        expression="q_sca",
        optimiser=torch.optim.SGD,
        lr_scheduler=None,
        lossFun=torch.nn.functional.mse_loss,
        max_iter=300):

        Losses = []

        for i, (r_c0, r_s0, n_c0, n_s0) in enumerate(
            zip(
                self.core_radius,
                self.shell_radius,
                self.core_refractiveIndex,
                self.shell_refractiveIndex,
            )
        ):
            print("Starting run {}.".format(i + 1))
            # k0 = 2 * torch.pi / torch.linspace(400, 800, 100)

            # - initial guess
            r_c = torch.tensor(r_c0, requires_grad=True)
            r_s = torch.tensor(r_s0, requires_grad=True)
            n_c = torch.tensor(n_c0, requires_grad=True)
            n_s = torch.tensor(n_s0, requires_grad=True)

            # print("init.:", [f"{d.detach().numpy():.3f}" for d in [r_c, r_s, n_c, n_s]])

            # - optimization loop
            optimizer = torch.optim.Adam([r_c, r_s, n_c, n_s], lr=0.1)

            losses = []

            for o in range(max_iter + 1):
                optimizer.zero_grad()

                args = (k0, r_c, n_c**2, r_s, n_s**2)

                iteration_n = farfield.cross_sections(*args)[expression]

                loss = lossFun(target, iteration_n)

                losses.append(loss.detach().item())

                loss.backward(retain_graph=False)
                optimizer.step()
                if o % 50 == 0:
                    print(o, loss.item())

            self.core_radius[i] = r_c  # nm
            self.shell_radius[i] = r_s  # nm
            self.core_refractiveIndex[i] = n_c
            self.shell_refractiveIndex[i] = n_s

            Losses.append(losses)
            print("Run {} completed!".format(i + 1))
        return Losses


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # define range of starting parameter combinations
    r_c_min, r_c_max = 10.0, 20.0
    r_s_min, r_s_max = 45.0, 55.0
    n_c_min, n_c_max = 2.0 + 0.1j, 2.0 + 0.1j
    n_s_min, n_s_max = 5.0 + 0.2j, 5.0 + 0.2j
    # define number of starting parameter combinations
    NumComb = 10

    r_c0, r_s0, n_c0, n_s0 = seedComb(
        r_c_min,
        r_c_max,
        r_s_min,
        r_s_max,
        n_c_min,
        n_c_max,
        n_s_min,
        n_s_max,
        NumComb=NumComb,
    )

    test_particle = particle(
        r_c=r_c0,
        r_s=r_s0,
        n_c=n_c0,
        n_s=n_s0,
    )

    starting_wavelength = 200  # nm
    ending_wavelength = 600  # nm

    N_pt_test = 200
    k0 = (
        2 * torch.pi / torch.linspace(starting_wavelength, ending_wavelength, N_pt_test, dtype=torch.double)
    )

    res_cs = farfield.cross_sections(
        k0=k0,
        r_c=30.0,
        eps_c=(4.0 + 0.1j) ** 2,
        r_s=50.0,
        eps_s=(3.0 + 0.1j) ** 2,
        eps_env=1,
        n_max=8,
    )

    target = res_cs["q_sca"]

    # print(target)

    LossCurves = test_particle.optimise(k0, target, max_iter=200)

    plt.plot(target.detach().numpy(), label = "Target", linestyle = "--", linewidth = 2.0)
    for i, (r_c0, r_s0, n_c0, n_s0) in enumerate(
        zip(
            test_particle.core_radius,
            test_particle.shell_radius,
            test_particle.core_refractiveIndex,
            test_particle.shell_refractiveIndex,
        )
    ):
        args = (k0, r_c0, n_c0**2, r_s0, n_s0**2)
        to_plot = farfield.cross_sections(*args)["q_sca"]
        plt.plot(to_plot, label = "Run {}".format(i))
    plt.legend()
    plt.show()



    for loss in LossCurves:
        plt.plot(loss)
        plt.yscale("log")
    plt.show()
