# -*- coding: utf-8 -*-
"""
main particle class
"""
import warnings

import torch


class Particle:
    def __init__(
        self, r_core, mat_core, r_shell=None, mat_shell=None, mat_env=1.0, device=None
    ):
        """Core-shell particle class

        High-level user interface, does not support multiple particles.
        To evaluate multiple particles at once directly use
        :func:`coreshell.cross_sections` or :func:`coreshell.angular_scattering`
        which support particle vectorisation.

        Args:
            r_core (float): core radius (in nm)
            mat_core (pymiediff material): core material. Either class for :mod:`pymiediff.materials` or float. In the case of a float, a constant material :class:`pymiediff.materials.MatConstant` will be created using the float as refractive index value.
            r_shell (float, optional): shell radius (in nm). If None, create homogeneous particle without shell. Defaults to None.
            mat_shell (pymiediff material, optional): Shell material. Defaults to None.
            mat_env (pymiediff material, optional): Environment material. Defaults to 1.0.
        """
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        if r_shell is None or mat_shell is None:
            assert (
                mat_shell is None
            ), "either both, or none of shell radius and material must be given."
            assert (
                r_shell is None
            ), "either both, or none of shell radius and material must be given."

        self.r_c = torch.as_tensor(r_core, device=self.device)  # core radius, nm
        if r_shell is not None:
            self.r_s = torch.as_tensor(r_shell, device=self.device)  # shell radius, nm
        else:
            self.r_s = None

        # create actual materials if float or int is given
        from pymiediff.materials import MatConstant

        if type(mat_core) in (float, int, complex, torch.Tensor):
            self.mat_c = MatConstant(mat_core**2, device=self.device)
        else:
            self.mat_c = mat_core
            self.mat_c.set_device(self.device)

        if mat_shell is not None:
            if type(mat_shell) in (float, int, complex, torch.Tensor):
                self.mat_s = MatConstant(mat_shell**2, device=self.device)
            else:
                self.mat_s = mat_shell
                self.mat_s.set_device(self.device)
        else:
            self.mat_s = None

        if type(mat_env) in (float, int, complex, torch.Tensor):
            self.mat_env = MatConstant(mat_env**2, device=self.device)
        else:
            self.mat_env = mat_env
            self.mat_env.set_device(self.device)

    def set_device(self, device):
        self.device = device

        self.r_c = self.r_c.to(device=self.device)
        self.r_s = self.r_s.to(device=self.device)

        self.mat_c.set_device(self.device)
        self.mat_s.set_device(self.device)
        self.mat_env.set_device(self.device)

    def __repr__(self):
        out_str = ""
        if self.r_s is None:
            out_str += "homogeneous particle (on device: {})\n".format(self.device)
            out_str += " - radius   = {}nm\n".format(self.r_c.data)
            out_str += " - material : {}\n".format(self.mat_c.__name__)
        else:
            out_str += "core-shell particle\n"
            out_str += " - core radius    = {}nm\n".format(self.r_c.data)
            out_str += " - shell radius   = {}nm\n".format(self.r_s.data)
            out_str += " - core material  : {}\n".format(self.mat_c.__name__)
            out_str += " - shell material : {}\n".format(self.mat_s.__name__)
        out_str += " - environment    : {}\n".format(self.mat_env.__name__)
        return out_str

    def get_material_permittivities(self, k0: torch.Tensor) -> tuple:
        """return spectral permittivities of core, shell and environment

        Args:
            k0 (torch.Tensor): tensor containing all evaluation wavenumbers

        Returns:
            tuple: tensors containing the spectral permittivities of core, shell and environment at all wavenumbers `k0`
        """
        k0 = torch.as_tensor(k0, device=self.device)
        wl0 = 2 * torch.pi / k0

        eps_c = self.mat_c.get_epsilon(wavelength=wl0)
        eps_env = self.mat_env.get_epsilon(wavelength=wl0)

        if self.mat_s is None:
            r_s = self.r_c
            eps_s = eps_c
        else:
            r_s = self.r_s
            eps_s = self.mat_s.get_epsilon(wavelength=wl0)

        return eps_c, eps_s, eps_env

    def get_mie_coefficients(
        self, k0: torch.Tensor, return_internal=False, **kwargs
    ) -> dict:
        """get farfield cross sections

        returns a dict that contains cross sections as well
        as efficiencies (scaled by the geometric cross sections)

        Note: Mie series truncation is done automatically using
        the Wiscomb criterion:
        Wiscombe, W. J. "Improved Mie scattering algorithms."
        Appl. Opt. 19.9, 1505-1509 (1980)

        kwargs are passed to :func:`pymiediff.coreshell.mie_coefficients`


        Results are retured as a dictionary with keys:
            - 'a_n' : external electric Mie coefficient
            - 'b_n' : external magnetic Mie coefficient
            - 'k0' : evaluation wavenumbers
            - 'k' : evaluation wavenumbers in host medium
            - 'n' : mie orders
            - 'n_max' : maximum mie order
            - 'r_c' : core radius
            - 'r_s' : shell radius
            - 'eps_c' : core permittivities
            - 'eps_s' : shell permittivities
            - 'eps_env' : environmental permittivity
            - 'n_c' : core refractive index
            - 'n_s' : shell refractive index
            - 'n_env' : environmental refractive index
        if kwarg `return_internal` is True, the returned dict contains also:
            - 'c_n' : internal magnetic Mie coefficient (core)
            - 'd_n' : internal electric Mie coefficient (core)
            - 'f_n' : internal magnetic Mie coefficient - first kind (shell)
            - 'g_n' : internal electric Mie coefficient - first kind (shell)
            - 'v_n' : internal magnetic Mie coefficient - second kind (shell)
            - 'w_n' : internal electric Mie coefficient - second kind (shell)

        Args:
            k0 (torch.Tensor): tensor containing all evaluation wavenumbers
            return_internal (bool, optional): If True, return also internal Mie cofficients.

        Returns:
            dict: dict containing Mie coefficients
        """
        from pymiediff.coreshell import mie_coefficients

        k0 = torch.as_tensor(k0, device=self.device)

        eps_c, eps_s, eps_env = self.get_material_permittivities(k0)
        r_s = self.r_c if (self.r_s is None) else self.r_s

        res = mie_coefficients(
            k0,
            r_c=self.r_c,
            r_s=r_s,
            eps_c=eps_c,
            eps_s=eps_s,
            eps_env=eps_env,
            **kwargs,
        )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        _squeeze_dimensions(res)  # in place

        return res

    def get_cross_sections(self, k0: torch.Tensor, **kwargs) -> dict:
        """get farfield cross sections

        returns a dict that contains cross sections as well
        as efficiencies (scaled by the geometric cross sections)

        Note: Mie series truncation is done automatically using
        the Wiscomb criterion:
        Wiscombe, W. J. "Improved Mie scattering algorithms."
        Appl. Opt. 19.9, 1505-1509 (1980)

        kwargs are passed to :func:`pymiediff.coreshell.cross_sections`

        Args:
            k0 (torch.Tensor): tensor containing all evaluation wavenumbers

        Returns:
            dict: dict containing all resulting spectra
        """
        from pymiediff.coreshell import cross_sections

        k0 = torch.as_tensor(k0, device=self.device)

        eps_c, eps_s, eps_env = self.get_material_permittivities(k0)
        r_s = self.r_c if (self.r_s is None) else self.r_s

        res = cross_sections(
            k0,
            r_c=self.r_c,
            r_s=r_s,
            eps_c=eps_c,
            eps_s=eps_s,
            eps_env=eps_env,
            **kwargs,
        )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res = _squeeze_dimensions(res)

        return res

    def get_angular_scattering(
        self, k0: torch.Tensor, theta: torch.Tensor, **kwargs
    ) -> dict:
        """get angular scattering

        kwargs are passed to :func:`pymiediff.coreshell.angular_scattering`

        Args:
            k0 (torch.Tensor): tensor containing all evaluation wavenumbers
            theta (torch.Tensor): tensor containing all evaluation angles (rad)

        Returns:
            dict: dict containing all angular scattering results for all wavenumbers and angles
        """
        from pymiediff.coreshell import angular_scattering

        k0 = torch.as_tensor(k0, device=self.device)
        theta = torch.as_tensor(theta, device=self.device)

        eps_c, eps_s, eps_env = self.get_material_permittivities(k0)
        r_s = self.r_c if (self.r_s is None) else self.r_s

        res_angSca = angular_scattering(
            k0=k0,
            theta=theta,
            r_c=self.r_c,
            r_s=r_s,
            eps_c=eps_c,
            eps_s=eps_s,
            eps_env=eps_env,
            **kwargs,
        )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res_angSca = _squeeze_dimensions(res_angSca)

        return res_angSca

    def get_nearfields(self, k0: torch.Tensor, r_probe: torch.Tensor, **kwargs) -> dict:
        """get scattered (near)fields E_s and H_s at all positions `r_probe`

        kwargs are passed to :func:`pymiediff.coreshell.nearfields`.
        Illumination amplitude is set to E_0=1.

        Args:
            k0 (torch.Tensor): tensor containing all evaluation wavenumbers
            r_probe (torch.Tensor): tensor containing all Cartesian positions to evaluate. Shape (..., 3).

        Returns:
            dict: dict containing all scattered fields "E_s" and "H_s"
        """
        from pymiediff.coreshell import nearfields

        k0 = torch.as_tensor(k0, device=self.device)
        r_probe = torch.as_tensor(r_probe, device=self.device)
        assert r_probe.shape[-1] == 3

        eps_c, eps_s, eps_env = self.get_material_permittivities(k0)
        r_s = self.r_c if (self.r_s is None) else self.r_s

        res_nf = nearfields(
            k0=k0,
            r_probe=r_probe,
            r_c=self.r_c,
            r_s=r_s,
            eps_c=eps_c,
            eps_s=eps_s,
            eps_env=eps_env,
            **kwargs,
        )

        # single particle: remove empty dimension
        from pymiediff.helper.helper import _squeeze_dimensions

        res_nf = _squeeze_dimensions(res_nf)

        return res_nf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import pymiediff as pmd

    # - config
    wl0 = torch.linspace(500, 1000, 50)
    k0 = 2 * torch.pi / wl0

    r_core = 70.0
    r_shell = 100.0
    mat_core = pmd.materials.MatDatabase("Si")
    mat_shell = pmd.materials.MatDatabase("Ge")
    n_env = 1.0

    # - setup the particle
    p = Particle(
        r_core=r_core,
        r_shell=r_shell,
        mat_core=mat_core,
        mat_shell=mat_shell,
        mat_env=n_env,
    )
    print(p)

    # - efficiency spectra
    cs = p.get_cross_sections(k0)
    plt.figure()
    plt.plot(cs["wavelength"], cs["q_ext"], label="$Q_{ext}$")
    plt.plot(cs["wavelength"], cs["q_sca"], label="$Q_{sca}$")
    plt.plot(cs["wavelength"], cs["q_abs"], label="$Q_{abs}$")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # - scattering radiation pattern
    theta = torch.linspace(0.0, 2 * torch.pi, 100)
    angular = p.get_angular_scattering(k0, theta)

    plt.figure(figsize=(12, 2))
    for i, i_k0 in enumerate(range(len(k0))[::5]):
        ax = plt.subplot(1, 10, i + 1, polar=True)
        plt.title(f"{wl0[i_k0]:.1f} nm")
        ax.plot(angular["theta"], angular["i_unpol"][i_k0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    plt.show()
