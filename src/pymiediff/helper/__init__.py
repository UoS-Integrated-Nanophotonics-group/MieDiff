# -*- coding: utf-8 -*-
"""Helper namespace for :mod:`pymiediff`.

This package groups utility functions used by the main scattering solvers and
exposes an optional bridge to :mod:`torchgdm`.

Submodules
----------
.. currentmodule:: pymiediff.helper

.. autosummary::
   :toctree: generated/

   helper
   tg

Frequently used symbols
-----------------------
.. autosummary::
   :toctree: generated/

   helper.get_truncution_criteroin_wiscombe
   helper.get_truncution_criteroin_pena2009
   helper.detach_tensor
   helper.transform_fields_spherical_to_cartesian
   helper.transform_spherical_to_xyz
   helper.transform_xyz_to_spherical
   helper.interp1d
   helper.funct_grad_checker
   helper.plane_wave_expansion
   tg.StructAutodiffMieEffPola3D
   tg.StructAutodiffMieGPM3D
"""
from .helper import detach_tensor
from .helper import plane_wave_expansion
from .helper import transform_fields_spherical_to_cartesian
from .helper import transform_spherical_to_xyz
from .helper import transform_xyz_to_spherical
from .helper import funct_grad_checker
from .helper import num_center_diff
from .helper import interp1d
from .helper import get_truncution_criteroin_wiscombe
from .helper import get_truncution_criteroin_pena2009

from . import tg
from . import helper
