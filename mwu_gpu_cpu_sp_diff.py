#!/usr/bin/env python

"""
Perform the Mann-Whitney U test in parallel for the following ensembles:
- GPU double precision (reference & control)
- CPU double precision
- GPU single precision
- GPU double precision with additional explicit diffusion

Make sure to compile the cpp file for the Mann-Whitney U test first before
running this script (see mannwhitneyu.cpp).

Copyright (c) 2021 ETH Zurich, Christian Zeman
MIT License
"""

import numpy as np
import xarray as xr
import pickle
import mannwhitneyu as mwu
from joblib import Parallel, delayed

rpert = 'e4'        # prefix
n_runs = 50         # total number of runs
n_sel = 100         # how many times we randomly select runs
alpha = 0.05        # significance level
nm = 20             # members per ensemble
u_crit = 127        # nm = 20
rej_rates = {}      # results
replace = False     # to bootstrap or not to bootstrap

# Variables
variables = ['t_850hPa', 'fi_500hPa', 'u_10m', 't_2m', 'precip', 'asob_t',
             'athb_t', 'ps']
# Please note that this script will create one task for each variable in
# order to run the calculation for each variable on a dedicated core
# (see below).

path_gpu = '../data/10d_gpu_cpu_sp_diff/gpu_dycore/'
path_cpu = '../data/10d_gpu_cpu_sp_diff/cpu_nodycore/'
path_gpu_sp = '../data/10d_gpu_cpu_sp_diff/gpu_dycore_sp/'
path_gpu_diff = '../data/10d_gpu_cpu_sp_diff/gpu_dycore_diff/'
rej_rates['c'] = {}
rej_rates['cpu'] = {}
rej_rates['sp'] = {}
rej_rates['diff'] = {}

runs_r = {}
runs_c = {}
runs_cpu = {}
runs_sp = {}
runs_diff = {}

# Load data for gpu (reference and control) and cpu
for i in range(n_runs):
    i_str_r = str(i).zfill(4)
    i_str_c = str(i+n_runs).zfill(4)
    fname_r = path_gpu + rpert + '_' + i_str_r + '.nc'
    fname_c = path_gpu + rpert + '_' + i_str_c + '.nc'
    fname_cpu = path_cpu + rpert + '_' + i_str_r + '.nc'
    fname_sp = path_gpu_sp + rpert + '_' + i_str_r + '.nc'
    fname_diff = path_gpu_diff + rpert + '_' + i_str_r + '.nc'
    runs_r[i] = {}
    runs_c[i] = {}
    runs_cpu[i] = {}
    runs_sp[i] = {}
    runs_diff[i] = {}
    runs_r[i]['dset'] = xr.open_dataset(fname_r)
    runs_c[i]['dset'] = xr.open_dataset(fname_c)
    runs_cpu[i]['dset'] = xr.open_dataset(fname_cpu)
    runs_sp[i]['dset'] = xr.open_dataset(fname_sp)
    runs_diff[i]['dset'] = xr.open_dataset(fname_diff)


# Function to be run in parallel
def get_rrate(vname, runs_r, runs_c, runs_cpu, runs_sp, runs_diff,
              n_sel, n_runs, nm, u_crit):

    rr = {}
    rr['c'] = {}
    rr['cpu'] = {}
    rr['sp'] = {}
    rr['diff'] = {}

    # initialize arrays
    nt, ny, nx = runs_r[0]['dset'][vname].shape
    values_r = np.zeros((nt, ny, nx, nm))
    values_c = np.zeros((nt, ny, nx, nm))
    values_cpu = np.zeros((nt, ny, nx, nm))
    values_sp = np.zeros((nt, ny, nx, nm))
    values_diff = np.zeros((nt, ny, nx, nm))
    results_c = np.zeros((n_sel, nt))
    results_cpu = np.zeros((n_sel, nt))
    results_sp = np.zeros((n_sel, nt))
    results_diff = np.zeros((n_sel, nt))
    spatial_c = np.zeros((n_sel, nt, ny, nx))
    spatial_cpu = np.zeros((n_sel, nt, ny, nx))
    spatial_sp = np.zeros((n_sel, nt, ny, nx))
    spatial_diff = np.zeros((n_sel, nt, ny, nx))

    # Do test multiple times with random selection of ensemble members
    for s in range(n_sel):
        if ((s+1) % 10 == 0):
            print(vname + ': ' + str(s+1) + " / " + str(n_sel))

        # Pick random samples for comparison
        idxs = np.random.choice(np.arange(n_runs), nm, replace=replace)

        # Put together arrays
        for i in range(nm):
            values_r[:,:,:,i] = runs_r[idxs[i]]['dset'][vname].values
            values_c[:,:,:,i] = runs_c[idxs[i]]['dset'][vname].values
            values_cpu[:,:,:,i] = runs_cpu[idxs[i]]['dset'][vname].values
            values_sp[:,:,:,i] = runs_sp[idxs[i]]['dset'][vname].values
            values_diff[:,:,:,i] = runs_diff[idxs[i]]['dset'][vname].values

        # Call test
        below_c = mwu.mwu(values_r, values_c, u_crit)
        below_cpu = mwu.mwu(values_r, values_cpu, u_crit)
        below_sp = mwu.mwu(values_r, values_sp, u_crit)
        below_diff = mwu.mwu(values_r, values_diff, u_crit)
        results_c[s] = np.mean(below_c, axis=(1,2))
        results_cpu[s] = np.mean(below_cpu, axis=(1,2))
        results_sp[s] = np.mean(below_sp, axis=(1,2))
        results_diff[s] = np.mean(below_diff, axis=(1,2))
        spatial_c[s] = below_c
        spatial_cpu[s] = below_cpu
        spatial_sp[s] = below_sp
        spatial_diff[s] = below_diff

    # Store results
    rr['c'][vname] = {}
    rr['c'][vname]['q_05'] = np.quantile(results_c, 0.5, axis=0)
    rr['c'][vname]['q_005'] = np.quantile(results_c, 0.05, axis=0)
    rr['c'][vname]['q_095'] = np.quantile(results_c, 0.95, axis=0)
    rr['c'][vname]['mean'] = np.mean(results_c, axis=0)
    rr['c'][vname]['min'] = np.min(results_c, axis=0)
    rr['c'][vname]['max'] = np.max(results_c, axis=0)
    rr['c'][vname]['reject'] = results_c
    rr['c'][vname]['spatial'] = spatial_c

    rr['cpu'][vname] = {}
    rr['cpu'][vname]['q_05'] = np.quantile(results_cpu, 0.5, axis=0)
    rr['cpu'][vname]['q_005'] = np.quantile(results_cpu, 0.05, axis=0)
    rr['cpu'][vname]['q_095'] = np.quantile(results_cpu, 0.95, axis=0)
    rr['cpu'][vname]['mean'] = np.mean(results_cpu, axis=0)
    rr['cpu'][vname]['min'] = np.min(results_cpu, axis=0)
    rr['cpu'][vname]['max'] = np.max(results_cpu, axis=0)
    rr['cpu'][vname]['reject'] = results_cpu
    rr['cpu'][vname]['spatial'] = spatial_cpu

    rr['sp'][vname] = {}
    rr['sp'][vname]['q_05'] = np.quantile(results_sp, 0.5, axis=0)
    rr['sp'][vname]['q_005'] = np.quantile(results_sp, 0.05, axis=0)
    rr['sp'][vname]['q_095'] = np.quantile(results_sp, 0.95, axis=0)
    rr['sp'][vname]['mean'] = np.mean(results_sp, axis=0)
    rr['sp'][vname]['min'] = np.min(results_sp, axis=0)
    rr['sp'][vname]['max'] = np.max(results_sp, axis=0)
    rr['sp'][vname]['reject'] = results_sp
    rr['sp'][vname]['spatial'] = spatial_sp

    rr['diff'][vname] = {}
    rr['diff'][vname]['q_05'] = np.quantile(results_diff, 0.5, axis=0)
    rr['diff'][vname]['q_005'] = np.quantile(results_diff, 0.05, axis=0)
    rr['diff'][vname]['q_095'] = np.quantile(results_diff, 0.95, axis=0)
    rr['diff'][vname]['mean'] = np.mean(results_diff, axis=0)
    rr['diff'][vname]['min'] = np.min(results_diff, axis=0)
    rr['diff'][vname]['max'] = np.max(results_diff, axis=0)
    rr['diff'][vname]['reject'] = results_diff
    rr['diff'][vname]['spatial'] = spatial_diff

    return rr

# Run in parallel
n_vars = len(variables)
rr_all = Parallel(n_jobs=n_vars)(delayed(get_rrate)
                    (vname, runs_r, runs_c, runs_cpu, runs_sp, runs_diff,
                     n_sel, n_runs, nm, u_crit)
                     for vname in variables)

# Concatenate dictionaries
rej_rates = {}
rej_rates['c'] = {}
rej_rates['cpu'] = {}
rej_rates['sp'] = {}
rej_rates['diff'] = {}
i = 0
for vname in variables:
    rej_rates['c'][vname] = {}
    rej_rates['cpu'][vname] = {}
    rej_rates['sp'][vname] = {}
    rej_rates['diff'][vname] = {}
    rej_rates['c'][vname].update(rr_all[i]['c'][vname])
    rej_rates['cpu'][vname].update(rr_all[i]['cpu'][vname])
    rej_rates['sp'][vname].update(rr_all[i]['sp'][vname])
    rej_rates['diff'][vname].update(rr_all[i]['diff'][vname])
    i += 1

# Save rejection rates
with open('rr_mannwhitneyu.pickle', 'wb') as handle:
    pickle.dump(rej_rates, handle)
