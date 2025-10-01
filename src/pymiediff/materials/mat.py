# -*- coding: utf-8 -*-
"""
provider for dielectric functions of selected materials.
"""
# %%

import warnings
import importlib.resources as pkg_resources
import pathlib

import torch

from pymiediff.helper import interp1d
from pymiediff import materials  # caution, dangerous wrt circular imports


DTYPE_FLOAT = torch.float32
DTYPE_COMPLEX = torch.complex64


# --- get all available tabulated materials
DATA_FOLDER = "data/"
data_files = pkg_resources.files(materials).joinpath(DATA_FOLDER).iterdir()

REFINDEX_DATA = {}
for f in data_files:
    f_n = pathlib.Path(f).name
    mat_name = f_n.split("_")[0]
    REFINDEX_DATA[mat_name.lower()] = [f, mat_name]


def list_available_materials(verbose=False):
    """Return all keys for the database materials"""
    if verbose:
        for f in REFINDEX_DATA:
            print("{}: ".format(f, pathlib.Path(REFINDEX_DATA[f]).name))
    return [f for f in REFINDEX_DATA]


# --- internal helpers
def _load_tabulated(dat_str):
    rows = dat_str["data"].split("\n")
    splitrows = [c.split() for c in rows]
    wl = []
    eps = []
    for s in splitrows:
        if len(s) > 0:
            wl.append(1000.0 * float(s[0]))  # microns --> nm
            _n = float(s[1])
            if len(s) > 2:
                _k = float(s[2])
            else:
                _k = 0.0
            eps.append((_n + 1j * _k) ** 2)
    return wl, eps


def _load_formula(dat_str):
    model_type = int((dat_str["type"].split())[1])
    coeff = [float(s) for s in dat_str["coefficients"].split()]
    for k in ["range", "wavelength_range"]:
        if k in dat_str:
            break
    # validity range (convert to nm)
    wl_range = [1e3 * float(dat_str[k].split()[0]), 1e3 * float(dat_str[k].split()[1])]

    return model_type, wl_range, coeff


# --- material defining base class
class MaterialBase:
    """base class for material permittivity"""

    __name__ = "material dielectric constant base class"

    def __init__(self, device: torch.device = "cpu"):
        self.device = device

    def __repr__(self, verbose: bool = False):
        """description about material"""
        out_str = " ------ base material class - doesn't define anything yet -------"
        return out_str

    def set_device(self, device):
        """move all tensors of the class to device"""
        self.device = device

    def get_epsilon(self, wavelength: float):
        """return permittivity at `wavelength`"""
        raise NotImplementedError("Needs to be implemented in child class.")

    def plot_epsilon(self, wavelengths=torch.linspace(400, 1400, 100)):
        """plot the permittivity dispersion

        Args:
            wavelengths (torch.Tensor, optional): wavelengths to evaluate. Defaults to torch.linspace(400, 1400, 100).
            tensor_comp (list, optional): permittivity tensor component indices. Defaults to [0, 0].
        """
        import matplotlib.pyplot as plt
        from pymiediff.helper.plotting import _get_axis_existing_or_new_axes
        from pymiediff.helper import detach_tensor

        eps = self.get_epsilon(wavelengths)

        # plot
        ax, show = _get_axis_existing_or_new_axes()

        plt.title("epsilon - '{}'".format(self.__name__))
        plt.plot(
            detach_tensor(wavelengths),
            detach_tensor(eps.real),
            label=r"Re($\epsilon$)",
        )
        plt.plot(
            detach_tensor(wavelengths),
            detach_tensor(eps.imag),
            label=r"Im($\epsilon$)",
        )
        plt.legend()
        plt.xlabel("wavelength (nm)")
        plt.ylabel("permittivity")

        if show:
            plt.show()

    def plot_refractive_index(self, wavelengths=torch.linspace(400, 1400, 100)):
        """plot the refractive index dispersion

        Args:
            wavelengths (torch.Tensor, optional): wavelengths to evaluate. Defaults to torch.linspace(400, 1400, 100).
            tensor_comp (list, optional): refractive index tensor component indices. Defaults to [0, 0].
        """
        import matplotlib.pyplot as plt
        from pymiediff.helper.plotting import _get_axis_existing_or_new_axes
        from pymiediff.helper import detach_tensor
        

        n_mat = self.get_epsilon(wavelengths) ** 0.5

        # plot
        ax, show = _get_axis_existing_or_new_axes()

        plt.title("ref. index - '{}'".format(self.__name__))
        plt.plot(
            detach_tensor(wavelengths),
            detach_tensor(n_mat.real),
            label="n",
        )
        plt.plot(
            detach_tensor(wavelengths),
            detach_tensor(n_mat.imag),
            label="k",
        )
        plt.legend()
        plt.xlabel("wavelength (nm)")
        plt.ylabel("refractive index")

        if show:
            plt.show()


class MatConstant(MaterialBase):
    """constant material index

    Material without dispersion
    """

    def __init__(self, eps=2.0 + 0.0j, device: torch.device = "cpu"):
        """constant permittivity material

        Args:
            eps (complex, optional): complex permittivity value. Defaults to (2.0 + 0.0j).
            device (torch.device, optional): Defaults to "cpu"
        """
        super().__init__(device=device)

        self.eps_scalar = torch.as_tensor(eps, dtype=DTYPE_COMPLEX, device=self.device)

        _eps_re = torch.round(self.eps_scalar.real, decimals=2)
        _eps_im = torch.round(self.eps_scalar.imag, decimals=3)
        if _eps_im == 0:
            self.__name__ = "eps={:.2f}".format(_eps_re)
        else:
            self.__name__ = "eps={:.2f}+i{:.3f}".format(_eps_re, _eps_im)

    def __repr__(self, verbose: bool = False):
        """description about material"""
        out_str = "constant, isotropic material. permittivity = {:.2f}".format(
            self.eps_scalar
        )
        return out_str

    def set_device(self, device):
        super().set_device(device)
        self.eps_scalar = self.eps_scalar.to(device)

    def get_epsilon(self, wavelength):
        """dispersionless, constant permittivity function

        Args:
            wavelength (float): in nm

        Returns:
            torch.Tensor: (3,3) complex permittivity tensor at `wavelength`
        """
        ones = 1.0
        wavelength = torch.as_tensor(wavelength, device=self.device)
        wavelength = wavelength.squeeze()

        # multiple wavelengths
        if len(wavelength.shape) == 1:
            ones = torch.ones(wavelength.shape[0], device=self.device)

        return ones * self.eps_scalar


# --- main interface classes for tabulated permittivity
class MatDatabase(MaterialBase):
    """dispersion from a database entry

    Use permittivity data from included database (data from https://refractiveindex.info/),
    or by loading a yaml file downloaded from https://refractiveindex.info/. Currently
    supported ref.index formats are tabulated n(k) data or Sellmeier model.

    Tabulated materials natively available in pymiediff can be
    printed via :func:`pymiediff.materials.list_available_materials()`

    Requires `pyyaml` (pip3 install pyyaml)

    Parameters
    ----------
    name : str
        name of database entry

    yaml_file : str, default: None
        optional filename of yaml refractiveindex data to load. In case a
        filename is provided, `name` will only be used as __name__ attribute
        for the class instance.

    """

    def __init__(
        self,
        name="",
        yaml_file=None,
        device: torch.device = "cpu",
        init_lookup_wavelengths=None,
    ):
        """dispersion from a database entry

        supports data following the yaml format of refractiveindex.info.
        Currently tabulated permittivity and Sellmeier models are supported.

        Args:
            name (str, optional): Name of database entry. Defaults to "".
            yaml_file (_type_, optional): path to optional yaml file with material data to load. If given, file is loaded and no data-base entry will be used, even if the name matches. Defaults to None.
            device (torch.device, optional): Defaults to "cpu".
            init_lookup_wavelengths (torch.Tensor, optional): optional list of wavelengths to generate an initial lookup table. Defaults to None.

        Raises:
            ValueError: Unknown material or unknown dispersion model type
        """
        import yaml

        super().__init__(device=device)

        if (name == "") and (yaml_file is None):
            print("No material specified. Available materials in database: ")
            for k in REFINDEX_DATA:
                print("     - '{}'".format(k))
            del self
            return
        if (yaml_file is None) and (name.lower() not in REFINDEX_DATA):
            raise ValueError(
                "'{}': Unknown material. Available materials in database: {}".format(
                    name, REFINDEX_DATA.keys()
                )
            )

        # load database entry from yaml
        if yaml_file is None:
            yaml_file = REFINDEX_DATA[name.lower()][0]
            self.__name__ = REFINDEX_DATA[name.lower()][1]
        else:
            if name:
                self.__name__ = name
            else:
                self.__name__ = pathlib.Path(yaml_file).stem

        with open(yaml_file, "r", encoding="utf8") as f:
            self.dset = yaml.load(f, Loader=yaml.BaseLoader)

        if len(self.dset["DATA"]) > 1:
            warnings.warn(
                "Several model entries in data-set for '{}' ({}). Using first entry.".format(
                    name, yaml_file
                )
            )
        dat = self.dset["DATA"][0]
        self.type = dat["type"]
        self.wl_dat = torch.Tensor([])
        self.eps_dat = torch.Tensor([])
        self.lookup_eps = {}

        # load refractive index model.
        # currently supported: tabulated data and Sellmeier model.
        # - tabulated data
        if self.type.split()[0] == "tabulated":
            wl_dat, eps_dat = _load_tabulated(dat)
            self.wl_dat = torch.as_tensor(wl_dat, dtype=DTYPE_FLOAT, device=self.device)
            self.eps_dat = torch.as_tensor(
                eps_dat, dtype=DTYPE_COMPLEX, device=self.device
            )
            self.model_type = "data"
            self.coeff = []
            self.wl_range = [torch.min(self.wl_dat), torch.max(self.wl_dat)]

        # - Sellmeier
        elif self.type.split()[0] == "formula":
            self.model_type, self.wl_range, self.coeff = _load_formula(dat)
            if self.model_type == 1:
                self.model_type = "sellmeier"
        else:
            raise ValueError(
                "refractiveindex.info data type '{}' not implemented yet.".format(
                    self.type
                )
            )

        # optionally initialize wavelength lookup
        if init_lookup_wavelengths is not None:
            for wl in init_lookup_wavelengths:
                _eps = self._get_eps_single_wl(wl)

    def __repr__(self, verbose: bool = False):
        """description about material"""
        out_str = ' ----- Material "{}" ({}) -----'.format(
            self.__name__, self.model_type
        )
        if self.model_type == "data":
            out_str += "\n tabulated wavelength range: {:.1f}nm - {:.1f}nm".format(
                *self.wl_range
            )
        elif self.model_type == "sellmeier":
            out_str += "\n Sellmeier model validity range: {:.1f}nm - {:.1f}nm".format(
                *self.wl_range
            )
        return out_str

    def set_device(self, device):
        super().set_device(device)
        self.wl_dat = self.wl_dat.to(device)
        self.eps_dat = self.eps_dat.to(device)

        # transfer the lookup table
        for _w in self.lookup_eps:
            self.lookup_eps[_w] = self.lookup_eps[_w].to(device)

    def _eval(self, wavelength):
        """evaluate refractiveindex.info model"""

        # - tabulated, using bilinear interpolation
        if self.model_type == "data":
            # torch implementation of 1d interpolation:
            eps = interp1d(wavelength, self.wl_dat, self.eps_dat)

        # - Sellmeier
        elif self.model_type == "sellmeier":
            eps = 1 + self.coeff[0]

            def g(c1, c2, w):
                return c1 * (w**2) / (w**2 - c2**2)

            for i in range(1, len(self.coeff), 2):
                # wavelength factor 1/1000: nm --> microns
                wl_mu = wavelength / 1000.0
                eps += g(self.coeff[i], self.coeff[i + 1], wl_mu)

        else:
            raise ValueError(
                "Only formula '1' (Sellmeier) or 'data' models supported so far."
            )

        return eps

    def _get_eps_single_wl(self, wavelength):
        # memoize evaluations
        wl_key = float(wavelength)

        if wl_key in self.lookup_eps:
            eps = self.lookup_eps[wl_key]
        else:
            _eps = self._eval(wavelength)
            eps = torch.as_tensor(_eps, dtype=DTYPE_COMPLEX, device=self.device)
            self.lookup_eps[wl_key] = eps

        return eps

    def get_epsilon(self, wavelength):
        """get permittivity at `wavelength`

        Args:
            wavelength (float): in nm

        Returns:
            torch.Tensor: complex permittivity tensor at `wavelength`
        """
        wavelength = torch.as_tensor(wavelength, dtype=DTYPE_FLOAT, device=self.device)
        wavelength = wavelength.squeeze()
        
        # multiple wavelengths
        if len(torch.as_tensor(wavelength).shape) == 1:
            eps = torch.stack([self._get_eps_single_wl(wl) for wl in wavelength], dim=0)
        else:
            eps = self._get_eps_single_wl(wavelength)

        return eps
