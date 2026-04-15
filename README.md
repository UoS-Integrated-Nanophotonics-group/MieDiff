![pymiediff logo](https://uos-integrated-nanophotonics-group.github.io/MieDiff/_images/pymiediff_logo-dark.png)

# pyMieDiff
> pyTorch implementation of Mie theory

pyMieDiff is a [Mie scattering](https://en.wikipedia.org/wiki/Mie_scattering) toolkit for layered spherical (nano-)particles in a homogeneous, isotropic environment. It is a pure [PyTorch](https://pytorch.org/) implementation of stable, logarithmic derivative based Mie series (following [Peña and Pal, CPC 180, 2348 (2009)](https://doi.org/10.1016/j.cpc.2009.07.010)). 
The outstanding feature, compared to similar tools, is the general support of `torch`'s **automatic differentiation**, fully vectorization and full GPU compatibility.

The source code is available on the [github repository](https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff/). 
For details, please see the [online documentation](https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html).

If you use pyMieDiff for your projects, please cite [our paper (arxiv:2512.08614)](https://arxiv.org/abs/2512.08614).


## How to use

### Forward Mie evaluation:

```python
import torch
import pymiediff as pmd

# - setup the particle
mat_core = pmd.materials.MatDatabase("Si")
mat_shell = pmd.materials.MatDatabase("Ge")

p = pmd.Particle(
    r_layers=[50.0, 80.0],  # nm
    mat_layers=[mat_core, mat_shell],
)

# - calculate efficiencies / cross section spectra
wl = torch.linspace(500, 1000, 50)
cs = p.get_cross_sections(k0=2 * torch.pi / wl)

plt.plot(cs["wavelength"], cs["q_ext"], label="$Q_{ext}$")
```

### Autograd

PyMieDiff fully supports native torch autograd:

```python
# - gradient of scattering wrt wavelength
wl = torch.as_tensor(500.0)
wl.requires_grad = True
cs = p.get_cross_sections(k0=2 * torch.pi / wl)

cs["q_sca"].backward()
dQdWl = wl.grad
```


## Installing / Requirements


```shell
pip install pymiediff
```

pymiediff is tested under linux and windows with python versions 3.10 to 3.13. It requires following python packages

- **pytorch** (v2.0+)
- **scipy** (v1.9+)

Optional dependencies:

- **matplotlib** (plotting)
- **pyyaml** (tabulated permittivity data from refractiveindex.info)



### GPU support

Simply pass the `device` argument to the particle class:

```python
p = pmd.Particle(
    r_layers=[50.0, 80.0],  # nm
    mat_layers=[mat_core, mat_shell],
    device="cuda",
)
```


Note that GPU performance is heavily memory transfer bound, GPU starts to be of advantage only for several thousand concurrent vectorized evaluations.


## Features

List of features

* multilayer spherical particles
* scattering and extinction cross sections
* angular scattering
* scattered and internal near-fields
* pure python / pytorch
* full support of torch's automatic differentiation
* GPU support
* fully vectorized


## Package Layout

Main package incudes

* `Particle` class:
    * definition of multi-shell particles and high-level interface to main functionalities
* `multishell` submodule:
    * Mie coefficients and observables for multishell particles.
* `special` submodule:
    * Contains PyTorch compatible Spherical Bessel and Hankel functions and angular functions pi and tau.
* `materials` subpackage:
    * PyTorch compatible materials classes for permittivity interpolation, based on the [refractiveindex.info](https://refractiveindex.info) yaml format.


## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.


## Links

- documentation: https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html
- github repository: https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff


## Licensing

The code in this project is licensed under the [GNU GPLv3](http://www.gnu.org/licenses/).
