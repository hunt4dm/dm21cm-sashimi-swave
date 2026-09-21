# dm21cm-sashimi-swave

Research code for implementing spatially inhomogeneous **s-wave dark
matter annihilation** in [DM21cm](https://github.com/yitiansun/DM21cm),
including subhalo contributions based on the
[SASHIMI](https://github.com/shinichiroando/sashimi-c) framework.

This repository contains the implementation and precomputed tables used
to model s-wave dark matter annihilation in 21-cm cosmology. It is
intended as research code built on DM21cm for a specific analysis,
rather than as a standalone replacement for the original DM21cm package.

## Overview

The implementation follows the spatially inhomogeneous halo-based
treatment used in DM21cm for dark-matter annihilation, adapted to
velocity-independent s-wave annihilation.

-   `dm21cm/` contains the DM21cm source files used in this work
    together with the modifications required for s-wave annihilation.
-   `DM21cm_data/` contains the precomputed halo-model tables used by
    the s-wave implementation.
-   `scripts/` contains scripts used for the corresponding DM21cm
    simulations.

## Precomputed s-wave tables

Two precomputed tables are included:

-   `swave_hmf_summed_rate.h5` --- s-wave annihilation with smooth host
    halos only.
-   `swave_hmf_summed_rate_subhalo_sashimi.h5` --- s-wave annihilation
    including the SASHIMI subhalo contribution.

The table-building scripts are included in the DM21cm precomputation
directory.


## Main software used

This work builds on several public research packages. Users of this
repository should also consult the documentation, installation
instructions, and citation requirements of the original projects.


### DM21cm

[DM21cm](https://github.com/yitiansun/DM21cm) models spatially
inhomogeneous exotic energy injection and its impact on the 21-cm signal
by interfacing 21cmFAST with energy-deposition calculations based on
DarkHistory.


### SASHIMI

Subhalo properties and annihilation-luminosity enhancements are modeled
using the [SASHIMI](https://github.com/shinichiroando/sashimi-c)
framework.


### 21cmFAST

[21cmFAST](https://github.com/21cmfast/21cmFAST) provides the
semi-numerical high-redshift structure-formation, astrophysical, and
21-cm calculations used by DM21cm.


### DarkHistory

[DarkHistory](https://github.com/hongwanliu/DarkHistory) provides the
energy-deposition treatment for energetic photons and electrons used by
DM21cm.


## Installation and usage

This repository is not intended to replace the original DM21cm
installation. A working DM21cm environment and its required dependencies
should first be prepared following the instructions in the [original
DM21cm repository](https://github.com/yitiansun/DM21cm).

In particular, users should use the versions and configuration of
21cmFAST and DarkHistory required by DM21cm.

The relevant data and cache directories should be configured following
the original DM21cm setup. 


## Citation

## Citation

If you use this code, please cite the associated paper:
- arXiv:2609.21479
Please also cite the relevant original software and methodology papers listed
above, in particular the DM21cm and SASHIMI references.



## Acknowledgements

This work makes use of and builds upon the public
[DM21cm](https://github.com/yitiansun/DM21cm),
[SASHIMI](https://github.com/shinichiroando/sashimi-c),
[21cmFAST](https://github.com/21cmfast/21cmFAST), and
[DarkHistory](https://github.com/hongwanliu/DarkHistory) projects. We
thank their authors and contributors for making these research tools
publicly available.
