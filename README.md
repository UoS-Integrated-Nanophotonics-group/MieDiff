![pymiediff logo](https://uos-integrated-nanophotonics-group.github.io/MieDiff/_images/pymiediff_logo-dark.png)

# pyMieDiff
> pyTorch implementation of Mie theory

pyMieDiff is a [Mie scattering](https://en.wikipedia.org/wiki/Mie_scattering) toolkit for spherical core-shell (nano-)particles in a homogeneous, isotropic environment. It is implemented in [PyTorch](https://pytorch.org/). The outstanding feature compared to similar tools is the general support of `torch`'s **automatic differentiation**.

The source code is available on the [github repository](https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff/). 
For details, please see the [online documentation](https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html).

If you use pyMieDiff for your projects, please cite our paper (to be added):


## How to use

### Forward Mie evaluation:

```python
import torch
import pymiediff as pmd

# - setup the particle
mat_core = pmd.materials.MatDatabase("Si")
mat_shell = pmd.materials.MatDatabase("Ge")

p = pmd.Particle(
    r_core=50.0,  # nm
    r_shell=70.0,  # nm
    mat_core=mat_core,
    mat_shell=mat_shell,
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

Installation should work via pip on all major operating systems. For now please use the [github repository](https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff) as source, we will publish pymiediff on PyPi once all main features are implemented and tested.

```shell
pip install https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff/archive/refs/heads/main.zip
```

pymiediff is tested under linux and windows with python versions 3.9 to 3.12. It requires following python packages

- **pytorch** (v2.0+)
- **scipy** (v1.9+)

Optional dependencies:

- **matplotlib** (plotting)
- **pyyaml** (tabulated permittivity data from refractiveindex.info)



### GPU support

The native Bessel functions implemented in pyMieDiff support GPU. The computation device can be chosen by passing the "device" keyword argument to the particle class. Note that GPU performance is currently still slightly lower than CPU performance, due to memory transfer overhead. We plan to optimize this in the future.


## Features

List of features

* core-shell spherical particles
* multilayer spherical particles via `backend="pena"`
* pure python
* full support of torch's automatic differentiation
* GPU support
* fully vectorized

## Multilayer Backend (Peña/Yang Recurrence)

`pymiediff.coreshell.mie_coefficients` supports multilayer spheres through
`backend="pena"`:

```python
res = pmd.coreshell.mie_coefficients(
    k0=2 * torch.pi / wl,
    r_layers=torch.tensor([40.0, 70.0, 110.0]),  # nm
    eps_layers=torch.tensor([(2.2 + 0.1j) ** 2, 1.8**2, (1.5 + 0.05j) ** 2]),
    eps_env=1.0,
    backend="pena",
    n_max=20,
)
```

Notes:
- Existing `r_c/r_s/eps_c/eps_s` core-shell inputs still work unchanged.
- In this phase, `backend="pena"` implements external coefficients (`a_n`,
  `b_n`) only. `return_internal=True` raises `NotImplementedError`.

## Package Layout

Main package incudes

* `Particle` class:
    * definition of core-shell particles and interface to main functionalities
* `coreshell` submodule:
    * Mie coefficients and observables for coreshell particles.
* `special` submodule:
    * Contains PyTorch compatible Spherical Bessel and Hankel functions and angular functions pi and tau.


## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.


## Links

- documentation: https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html
- github repository: https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff


## Licensing

The code in this project is licensed under the [GNU GPLv3](http://www.gnu.org/licenses/).
