# -*- coding: utf-8 -*-
"""manage installation"""
from setuptools import setup, find_namespace_packages
import os
import re


# =============================================================================
# helper functions to extract meta-info from package
# =============================================================================
def read_version_file(*parts):
    return open(os.path.join(*parts), "r").read()


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def find_name(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__name__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find name string.")


def find_author(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find author string.")


# =============================================================================
# package module list
# =============================================================================
package_list = find_namespace_packages(where=".", include=["pymiediff*"])


# =============================================================================
# main setup
# =============================================================================
setup(
    name=find_name("pymiediff", "__init__.py"),
    version=find_version("pymiediff", "__init__.py"),
    author=find_author("pymiediff", "__init__.py"),
    author_email="O.K.Jackson@soton.ac.uk",
    description=(
        "An auto-diff enabled implementation of Mie theory via PyTorch."
    ),
    license="GPLv3+",
    long_description=read("README.md"),
    packages=package_list,
    # package_data={"pymiediff.materials.data": ["*.yml"]},  # maybe if we decide to package materials tabulated permittivity data
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Physics",
        "Environment :: Console",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Intended Audience :: Science/Research",
    ],
    url="https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff",
    download_url="",
    keywords=[
        "Mie Theory",
        "optical scattering",
        "nano optics",
        "automatic differentiation",
    ],
    install_requires=["torch>=2.0.0", "scipy>=1.10.0"],  # add here all required dependencies for automatic installation cia pip
    python_requires=">=3.9",
)
