![pymiediff logo](https://link.to.logo)

# pyMieDiff
> pyTorch implementation of Mie theory

pyMieDiff is a [Mie scattering](https://en.wikipedia.org/wiki/Mie_scattering) toolkit for spherical core-shell (nano-)particles in a homogeneous, isotropic environment. It is implemented in [PyTorch](https://pytorch.org/). The outstanding feature compared to similar tools is the general support of `torch`'s **automatic differentiation**.

If you use pyMieDiff for your projects, please cite our paper (to be added):


## Getting started

Some minimal example script

```python
import pymiediff

# --- todo

```


### GPU support not yet available

pyMieDiff currently does not provide GPU support, as it uses wrappers to scipy special functions. We plan to implement GPU-capable recurrence schemes for Bessel function evaluation in the future.


## Features

List of features

* pure python
* full support of torch's automatic differentiation
* core-shell spherical particles

## Package Layout

Main package incudes

* farfield submodule:
    * Contains functions to calulate farfield observables.
* coreshell submodule:
    * Contains Mie scattering coefficients for coreshell particles.
* special submodule:
    * Contains PyTorch compatible Spherical Bessel and Hankel functions and  angular functions pi and tau.


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


## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.


## Links

- documentation: TODO
- github repository: https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff


## Licensing

The code in this project is licensed under the [GNU GPLv3](http://www.gnu.org/licenses/).
