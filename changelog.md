# changelog

## [unreleased]
 - fix: GPM structures (`StructAutodiffMieGPM3D`, `extract_GPM_sphere_miediff`) failed with
   `ModuleNotFoundError` against torchgdm 0.58. A GPM is the global polarizability matrix: a
   few coupled dipoles that reproduce how a particle scatters. torchgdm moved its GPM tools
   from `torchgdm.struct.eff_model_tools` to `torchgdm.struct.gpm_tools`, and the bridge still
   asked for the old path. The bridge now looks in both places, so torchgdm 0.57 and 0.58 both
   work.
 - fix: `patch_torchgdm_autodiff()` silently did nothing against torchgdm 0.58. It patched
   `LinearSystemBase._get_full_Gdotalpha`, but that method now lives on
   `LinearSystemFullInverse`, so the subclass definition won and the in-place-operation fix
   never applied. The patch now finds the classes that define the method, skips any whose
   signature it cannot replace, and warns when it applies nothing.

## [v0.12]
 - fix: torchGDM effective model extraction using both s- and p- polarization 

## [v0.11]
 - new backend: new, logartihmic derivatives based, stable algorithm. 
 - N-layers support
 - N-layers nearfields based on logarithmic derivatives backend

## [v0.9]
 - bugfix for homogeneous spheres with some functions

## [v0.8] - 2025-11-14
 - multi-scattering simulations through torchGDM interface

## [v0.7] - 2025-10-31
 - nearfield calculations
 - internal refactoring: fully consistent vectorization conventions

## [v0.6] - 2025-09-23
 - full vectorization support
 - full documentation

## [v0.5] - 2025-06-12
 - GPU support for torch-backend

## [v0.4] - 2025-05-25
 - recurrence-based native torch implementation

## [v0.3] - 2025-05-09
 - significant performance optimizations

## [v0.2] - 2025-02-25
 - full refactoring of API

## [v0.1] - 2024-10-09
 - package setup - not fully working yet
