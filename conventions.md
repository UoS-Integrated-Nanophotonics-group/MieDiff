# conventions in pymiediff


## geometry / environment description

 - Use consistently as frequency input parameter: vacuum wave number, symbol `k0`
 - use radii as sphere size parameters
    - core radius symbol: `r_c`
    - shell radius symbol: `r_s`
 - use permittivity and permeability for material description
    - permittivity symbols (core, shell): `eps_c`, `eps_s`
    - permeability symbols (core, shell): `mu_c`, `mu_s`
    - environment description: `eps_env`, `mu_env` (purely real)

## Mie calculation

 - Mie order: `n` (`n_max` for setting the truncation limit)
 - efficiencies are given by lowercase `q`, followed by underline + 3-letter name: 
    - `q_sca`
    - `q_ext`
    - `q_abs`
 - cross sections similarly, starting with `cs`:
    - `cs_sca`
    - `cs_ext`
    - `cs_abs`
    - `cs_geo` (geometric cross section: footprint circle area)



## vectorization
 - first dimension: Mie order `n`
 - second dimension: particles
 - third dimension: wavelength / wavenumber
 - fourth dim.: other like angles, positions


# checklist: upload to pypi:

- change version number
- run tests
- commit changes
- python3 -m build       (requires pip install build)
- python3 -m twine upload dist/*      (requires pip install twine)
