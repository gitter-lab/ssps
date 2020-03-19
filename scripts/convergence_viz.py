# convergence_viz.py
#
# Visualize convergence analysis results.

import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import script_util as su

import matplotlib as mpl
mpl.rc('font', family='serif')

dataset = snakemake.input[0].split("/")[-2].split("_")[:-1]

kvs = su.parse_path_kvs(snakemake.input[0])
#v = kvs["v"]
n_replicates = len(snakemake.input)

PSRF_THRESH = snakemake.config["dream_challenge"]["convergence"]["psrf_ub"]
N_EFF_THRESH = snakemake.config["dream_challenge"]["convergence"]["neff_per_chain"]*4

def is_converged(pair):
    if None in pair:
        return False
    elif pair[0] >= PSRF_THRESH and pair[1] < N_EFF_THRESH:
        return False
    else:
        return True


def count_nonconverged(all_diagnostics, t):
    n_nc = 0
    for diagnostic_vec in all_diagnostics:
        if not is_converged(diagnostic_vec[t]):
            n_nc += 1
    return n_nc



def plot_nonconverged(replicates, stop_points, out_file, ylabel="Non-converged quantities"):

    plt.figure()
    # build vectors of:
    nonconv_counts = []
    for t, sp in enumerate(stop_points):
        t_results = []
        for rep in replicates:
            # store number of non-converged edge indicators
            t_results.append(count_nonconverged(rep, t))
        nonconv_counts.append(t_results)
    
    meds = [np.median(res) for res in nonconv_counts]
    mins = [min(res) for res in nonconv_counts]
    maxes = [max(res) for res in nonconv_counts]
    
    # Plot med, min, max
    plt.plot(stop_points, [0 for sp in stop_points], color="blue", linestyle="--")
    plt.fill_between(stop_points, mins, maxes, color="grey", label="Range ({} replicates)".format(n_replicates))
    plt.plot(stop_points, meds, color="k", label="Median ({} replicates)".format(n_replicates))
    
    plt.legend()
    
    plt.xlim(stop_points[0],stop_points[-1])
    plt.xticks(stop_points, rotation=45.0)
    
    plt.ylabel(ylabel, family="serif")
    plt.xlabel("Sampling Chain Length", family="serif")
    
    #plt.title("MCMC Convergence: {} variables".format(v), family="serif")
    plt.title("MCMC Convergence: {}".format(dataset), family="serif")
    plt.tight_layout()
    #savefig
    plt.savefig(out_file, dpi=300)
    plt.close()

# Get the replicates' convergence statistics 
replicates = [su.extract_from_file(f) for f in snakemake.input]

lambda_replicates = [[diag_vec for diag_vec in rep["conv_stats"]["lambda"]] for rep in replicates]
parent_replicates = [[diag_vec for ps in rep["conv_stats"]["parent_sets"] for diag_vec in ps] for rep in replicates] 

stop_points = replicates[0]["stop_points"]
for rep in replicates:
    if len(rep["stop_points"]) < len(stop_points):
        stop_points = rep["stop_points"]


parent_fname = snakemake.output[0]
#lambda_fname = ".".join(parent_fname.split("."))[:-1] + "_lambda.png"
lambda_fname = parent_fname[:-4] + "_lambda.png"

plot_nonconverged(parent_replicates, stop_points, parent_fname, 
                  ylabel="Non-converged edge probabilities")
plot_nonconverged(lambda_replicates, stop_points, lambda_fname,
                  ylabel="Non-converged lambda variables")


