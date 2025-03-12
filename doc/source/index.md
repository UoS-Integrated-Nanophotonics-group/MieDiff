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

If you use pyMieDiff for your projects, please cite our paper (to be added):

```{note}
   If you use pyMieDiff for your work, please cite our paper: ({download}`bibtex <_downloads/_placeholder.empty>`):

   **AUTHORS**  
   *TITLE*  
   JOURNAL REF.  
   
   [DOI: XXX](https://doi.org/XXX), [arXiv:XXX](http://arxiv.org/abs/XXX)  
   [download](./_downloads/dummy.pdf)  
```

## Installation

Install via pip:

```bash
$ pip install pymiediff
```

## How to use

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

# - calculate cross section spectra
wl = torch.linspace(500, 1000, 50)
cs = p.get_cross_sections(k0=2 * torch.pi / wl)

plt.plot(cs["wavelength"], cs["q_ext"], label="$Q_{ext}$")
```

## GPU support

pyMieDiff currently does not provide GPU support, as it uses wrappers to scipy special functions. We plan to implement GPU-capable recurrence schemes for Bessel function evaluation in the future.



## pyMieDiff documentation

```{toctree}
:caption: 'Contents:'
:maxdepth: 1

auto_gallery/index
api
```
