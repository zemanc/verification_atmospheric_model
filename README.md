# Source code for verification of numerical weather and climate model executables

Source code for "An Ensemble-Based Statistical Methodology to Detect Differences
in Weather and Climate Model Executables"

The files here contain the source coude that has been used to calculate some
of the rejection rates presented in the paper "An Ensemble-Based Statistical
Methodology to Detect Differences in Weather and Climate Model Executables"
from Christian Zeman and Christoph Schär.

mwu_gpu_cp_sp_diff.py:
Calculate rejection rates using the Mann-Whitney U test on a grid-cell
level for ensembles that have been produced with COSMO GPU DP, COSMO CPU DP,
COSMO GPU SP, and COSMO GPU DP DIFF. This version runs in parallel with
one task for each variable.

mwu_ks_studt.py:
Calculate rejection rates using different local statistical tests on
a grid-cell level (Mann-Whitney U test, Kolmogorov-Smirnov test,
Student's t-test) for the same cases as above.

mwu_update.py:
Calculate rejection rates using the Mann-Whitney U test on a grid-cell
level for ensembles that have been produced with COSMO before and after
a major system update of the underlying supercomputer.

mannwhitneyu.cpp:
C++ implementation of the Mann-Whitney U test on a grid-cell level.

kolmogorov-smirnov.cpp:
C++ implementation of the Kolmogorov-Smirnov test on a grid-cell level.

The corresponding data is available under
https://doi.org/10.5281/zenodo.5106467
