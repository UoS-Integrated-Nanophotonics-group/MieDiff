# -*- coding: utf-8 -*-
"""Compatibility shim for the legacy ``pymiediff.coreshell`` module.

All functionality now lives in :mod:`pymiediff.multishell`.

This wrapper is only for legacy codes that require the coreshell 
module name, it may be removed in future versions.
"""

from .multishell import *  # noqa: F401,F403
