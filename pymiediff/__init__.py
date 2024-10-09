# encoding=utf-8
#
# Copyright (C) 2023-2024, Oscar K. Jackson
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
pymiediff - pytorch implementation of Mie theory for core-shell spherical particles.
"""

__name__ = "pymiediff"
__version__ = "0.1"
__date__ = "10/09/2024"  # MM/DD/YYY
__license__ = "GPL3"
__status__ = "alpha"

__copyright__ = "Copyright 2023-2024, Oscar K. Jackson"
__author__ = "Oscar K. Jackson"
__maintainer__ = "Oscar K. Jackson"
__email__ = "O.K.Jackson@soton.ac.uk"
# other contributors:
__credits__ = [
    "Peter R. Wiecha",
]


# --- populate namespace
# import here all modules, subpackages, functions, classes available from the package
# that should be available from the top-level of the package namespace

# modules
from . import main
from . import special
