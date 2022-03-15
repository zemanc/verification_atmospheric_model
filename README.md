# Source code for verification of numerical weather and climate model executables

The files here contain the source coude that has been used to calculate some
of the rejection rates presented in the paper "An Ensemble-Based Statistical
Methodology to Detect Differences in Weather and Climate Model Executables"
from Christian Zeman and Christoph Schär in Geoscientific Model Development (https://doi.org/10.5194/gmd-2021-248).

mwu_gpu_cp_sp_diff.py:
Calculate rejection rates using the Mann-Whitney U test on a grid-cell
level for ensembles that have been produced with COSMO GPU DP, COSMO CPU DP,
COSMO GPU SP, and COSMO GPU DP DIFF. This version runs in parallel on a
specified amount of cores (nprocs).

mwu_ks_studt.py:
Calculate rejection rates using different local statistical tests on
a grid-cell level (Mann-Whitney U test, Kolmogorov-Smirnov test,
Student's t-test) for the same cases as above.

mwu_update.py:
Calculate rejection rates using the Mann-Whitney U test on a grid-cell
level for ensembles that have been produced with COSMO before and after
a major system update of the underlying supercomputer.

fdr_test.py:
Calculate rejection rates using the Student's t-test where the approach in the
paper using subsampling is compared to the FDR approach for determining
field significance.

mannwhitneyu.cpp:
C++ implementation of the Mann-Whitney U test on a grid-cell level.

kolmogorov-smirnov.cpp:
C++ implementation of the Kolmogorov-Smirnov test on a grid-cell level.

The corresponding data is available in Zenodo.

First part: https://doi.org/10.5281/zenodo.6354200

Second part: https://doi.org/10.5281/zenodo.6355647

The latest release can be found here: 

[![DOI](https://zenodo.org/badge/386663542.svg)](https://zenodo.org/badge/latestdoi/386663542)


