# -*- coding: utf-8 -*-
"""
main routines of pymiediff
"""
import warnings

import numpy as np
import torch

# from . import special
from pymiediff import special  # use absolute package internal imports!
from pymiediff import coreshell as cs
from pymiediff import angular as ang

# define here functions / classes that should be provided by
# the `main` module of the package


class particle:
    def __init__(
            self,
            r_c0=50.0,
            r_s0=100.0,
            eps_c0=2.0 + 0.1j,
            eps_s0=5.0 + 0.1j,
            eps_env=1.0):

        self.core_radius = r_c0  # nm
        self.shell_radius = r_s0  # nm
        self.core_refractiveIndex = eps_c0**0.5
        self.shell_refractiveIndex = eps_s0**0.5

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
        max_iter=300,
    ):
        # k0 = 2 * torch.pi / torch.linspace(400, 800, 100)

        # - initial guess
        r_c = torch.tensor(self.core_radius, requires_grad=True)
        r_s = torch.tensor(self.shell_radius, requires_grad=True)
        n_c = torch.tensor(self.core_refractiveIndex, requires_grad=True)
        n_s = torch.tensor(self.shell_refractiveIndex, requires_grad=True)

        # print("init.:", [f"{d.detach().numpy():.3f}" for d in [r_c, r_s, n_c, n_s]])

        # - optimization loop
        optimizer = torch.optim.Adam([r_c, r_s, n_c, n_s], lr=0.1)

        losses = []


        if expression in cs.expressions:
            forward = cs.scs
        elif expression in ang.expressions:
            forward = ang.smat



        for i in range(max_iter + 1):
            optimizer.zero_grad()

            args = (k0, r_c, n_c**2, r_s, n_s**2)

            iteration_n = forward(*args)[expression]

            loss = lossFun(target, iteration_n)

            losses.append(loss.detach().item())

            loss.backward(retain_graph=False)
            optimizer.step()

        self.core_radius = r_c  # nm
        self.shell_radius = r_s  # nm
        self.core_refractiveIndex = n_c
        self.shell_refractiveIndex = n_s

        return dict(
            loss_curve=losses,
            optimised_params=[r_c, r_s, n_c, n_s],
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_particle = particle(r_c0=50.0,
                             r_s0=100.0,
                             eps_c0=2.0 + 0.1j,
                             eps_s0=5.0 + 0.1j
                             )

    optimiser_results = particle.optimise(
        expression="q_sca",
        optimiser=torch.optim.SGD,
        lr_scheduler=None,
        lossFun=torch.nn.functional.mse_loss,
        max_iter=300,
    )

    losses = optimiser_results["loss_curve"]
    optimised_params = optimiser_results["optimised_params"]
