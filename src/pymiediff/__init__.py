# encoding=utf-8
#
# Copyright (C) 2025, O. K. Jackson, P. R. Wiecha
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
pyMieDiff - Mie + auto-diff

Mie theory for core-shell particles, implemted in pytorch.

API
===

Particle class
--------------

The :class:`pymiediff.Particle` class describes core-shell particles
and provides high-level access to the main functionalities:


.. currentmodule:: pymiediff

.. autosummary::
   :recursive:
   :toctree: generated/

   Particle


Materials
----------

pymiediff contains pyTorch autodiff classes for tabulated material
permittivities, compatible with the refractiveindex.info format.

.. autosummary::
   :toctree: generated/

   materials


Farfield
--------------

farfield contains pyTorch autodiff funtions that can be directly
used to compute the farfield obserables.

.. currentmodule:: pymiediff

.. autosummary::
   :toctree: generated/

   farfield


Special
----------

pymiediff contains pyTorch autodiff comptible spherical bessel functions and
their derivatives.

.. autosummary::
   :toctree: generated/

   special


Core-shell
----------

This contains the core-shell scattering coefficients

.. autosummary::
   :toctree: generated/

   coreshell


Helper
------

pyMieDiff contains tools e.g. for truncation critera, interpolation,
numerical gradients or plotting.

.. autosummary::
   :toctree: generated/

   helper

"""

__name__ = "pymiediff"
__version__ = "0.4"
__date__ = "05/25/2025"  # MM/DD/YYY
__license__ = "GPL3"
__status__ = "alpha"

__copyright__ = "Copyright 2024-2025"
__author__ = "Peter R. Wiecha, Oscar K. Jackson"
__maintainer__ = "Peter R. Wiecha, Oscar K. Jackson"
__email__ = "pwiecha@laas.fr, O.K.Jackson@soton.ac.uk"
# other contributors:
__credits__ = []


# --- populate namespace
# import here all modules, subpackages, functions, classes available from the package
# that should be available from the top-level of the package namespace

# modules
from pymiediff.main import Particle
from . import special
from . import coreshell
from . import helper
from . import materials
from . import farfield
