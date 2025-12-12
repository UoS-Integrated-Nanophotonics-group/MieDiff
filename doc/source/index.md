% pymiediff-doc documentation master file, created by
% sphinx-quickstart on Wed Mar 12 10:38:15 2025.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

```{image} _static/pymiediff_logo-dark.png
:class: only-dark
```

```{image} _static/pymiediff_logo-light.png 
:class: only-light
```

```{eval-rst}
|
```

# Welcome to pyMieDiff
> pyTorch implementation of Mie theory

pyMieDiff is a [Mie scattering](https://en.wikipedia.org/wiki/Mie_scattering) toolkit for spherical core-shell (nano-)particles in a homogeneous, isotropic environment. It is implemented in [PyTorch](https://pytorch.org/). The outstanding feature compared to similar tools is the general support of `torch`'s **automatic differentiation**.

The source code is available on the [github repository](https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff/). 

If you use pyMieDiff for your projects, please cite our paper (to be added):

```{note}
   If you use pyMieDiff for your work, please cite our paper: ({download}`bibtex <_downloads/pymiediff.bib>`):

   **Oscar K. C. Jackson, Simone De Liberato, Otto L. Muskens, Peter R. Wiecha**  
   *PyMieDiff: A differentiable Mie scattering library*  
   [arXiv:2512.08614](http://arxiv.org/abs/2512.08614)  
   [download](./_downloads/2512.08614v1.pdf)  
```

## Installation

Install via pip:

```bash
$ pip install pymiediff
```


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

## GPU support

Simply pass the `device` argument to the particle class:

```python
p = pmd.Particle(
    r_core=50.0,  # nm
    r_shell=70.0,  # nm
    mat_core=mat_core,
    mat_shell=mat_shell,
    device="cuda",
)
```

## Links

- documentation: https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html
- github repository: https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff


## pyMieDiff documentation

```{toctree}
:caption: 'Contents:'
:maxdepth: 1

auto_gallery/index
api
```
