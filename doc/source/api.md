# API documentation

```{eval-rst}
.. automodule:: pymiediff
   :no-members:
   :no-inherited-members:
   :no-special-members:
```

## Multilayer Mie Coefficients

`pymiediff.multishell.mie_coefficients` now accepts optional multilayer inputs:

- `r_layers`: layer outer radii, shape `(L,)` or `(N_part, L)`
- `eps_layers`: layer permittivities, shape `(L,)`, `(N_part, L)`, `(L, N_k0)`,
  or `(N_part, L, N_k0)`
- `backend="pena"`: Peña/Pal (Yang-recursive) multilayer evaluation

The legacy core-shell arguments (`r_c`, `r_s`, `eps_c`, `eps_s`) remain
supported and map internally to the layer representation.
The legacy module name `pymiediff.coreshell` is kept as a compatibility
shim and re-exports `pymiediff.multishell`.

Current limitation:
- for `backend="pena"`, only external coefficients `a_n` and `b_n` are
  implemented in this phase; `return_internal=True` is not available yet.
