# FEniCSx_ii (FEniCSx trace)

FEniCSx_ii is an extension of FEniCSx that allows users to work with non-conforming 3D-1D meshes.
The core algorithm is based on the framework proposed by [Kuchta 2021](https://doi.org/10.1007/978-3-030-55874-1_63) {cite}`intro-Kuchta2021trace`
and implemented in [FEniCS_ii](https://github.com/MiroK/fenics_ii).

The new framework addresses the limitation of $\mathrm{FEniCS}\_{\mathrm{ii}}$ not being MPI compatible.
Currently, the new framework does not use {cite}`intro-Mardal2012` [cbc.block](https://github.com/blocknics/cbc.block), and instead
implements the _matrix-matrix_ products in a non-lazy fashion, and uses [PETSc Nest](https://petsc.org/release/manualpages/Mat/MATNEST/) matrices
to set up the blocked system.

Given a 3D domain with a function $u\in V(\Omega)$ and a $1D$ domain $\Gamma$.
We define a restriction operator $\Pi:V\mapsto L^2(\Gamma)$,
which becomes a central part of the variational formulation.

See for instance
[D'Angelo & Quarteroni, 2008](https://doi.org/10.1142/S0218202508003108) {cite}`intro-dangelo20083d1d`,
[Kuchta 2021](https://doi.org/10.1007/978-3-030-55874-1_63) {cite}`intro-Kuchta2021trace` or
[Masri, Kuchta & Riviere, 2024](https://doi.org/10.1137/23M1627390) {cite}`intro-masri2024coupled3d1d`.

Several (non-local) operators are implemented in {py:mod}`fenicsx_ii`:

- {py:class}`PointwiseTrace<fenicsx_ii.PointwiseTrace>`, the operator: $\Pi(u)(\hat x)=u(\hat x)$, $\hat x \in \Gamma$ .
- {py:class}`Circle<fenicsx_ii.Circle>`, the operator $\Pi(u)(\hat x)=\frac{1}{\vert P_R \vert}\int_{P_{R}(\Gamma(\hat x))}u~\mathrm{d}s$, where $P_R(\Gamma(\hat x))$ is the perimeter of a disk with radius $R$, normal aligning with $\Gamma(\hat x)$ and origin at $\hat x$.
- {py:class}`Disk<fenicsx_ii.Disk>`, the operator $\Pi(u)(\hat x)=\frac{1}{\vert D_R \vert}\int_{D_R(\Gamma(\hat x))} u~\mathrm{d}x$, where $D_R(\Gamma(\hat x))$ is the disk with radius $R$, normal aligining with $\Gamma(\hat x)$ and origin at $\hat x$.

Any other operator can be implemented by following the {py:class}`ReductionOperator<fenicsx_ii.ReductionOperator>`-protocol.

## Installation

Users are encouraged to install `fenicsx_ii` with `pip` in an environment where `dolfinx` is already installed.

### `pip`
To install the package with `pip` run

```bash
python3 -m pip install fenicsx_ii --no-build-isolation
```

Note that you should pass the flag `--no-build-isolation` to `pip` to avoid issues with the build environment, such as incompatible versions of `nanobind`.

## Funding

The development of FEniCSx_ii has been funded by the Wellcome Trust,
grant number: 313298/Z/24/Z

## References

```{bibliography}
:filter: cited
:labelprefix:
:keyprefix: intro-
```
