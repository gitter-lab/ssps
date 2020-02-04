# convergence_viz.py
#
# Visualize convergence analysis results.

import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import script_util as su

import matplotlib as mpl
mpl.rc('font', family='serif')

print("I'M THE INPUT: ", snakemake.input)
print("I'M THE OUTPUT: ", snakemake.output)

kvs = su.parse_path_kvs(snakemake.input[0])
v = kvs["v"]
n_replicates = len(snakemake.input)

PSRF_THRESH = snakemake.config["convergence_analysis"]["psrf_ub"]
N_EFF_THRESH = snakemake.config["convergence_analysis"]["neff_per_chain"]*4

def is_converged(pair):
    if None in pair:
        return False
    elif pair[0] >= PSRF_THRESH and pair[1] < N_EFF_THRESH:
        return False
    else:
        return True

def count_nonconverged(edge_indicators, t):
    n_nc = 0
    for ps in edge_indicators:
        for p in ps:
            if not is_converged(p[t]):
                n_nc += 1
    return n_nc


# Get the replicates' convergence statistics 
replicates = [su.extract_from_file(f) for f in snakemake.input]

stop_points = replicates[0]["stop_points"]
for rep in replicates:
    if len(rep["stop_points"]) < len(stop_points):
        stop_points = rep["stop_points"]


# build vectors of:
nonconv_counts = []
for t, sp in enumerate(stop_points):
    t_results = []
    for rep in replicates:
        # store number of non-converged edge indicators
        t_results.append(count_nonconverged(rep["conv_stats"]["parent_sets"], t))
    nonconv_counts.append(t_results)

means = [sum(res)/len(res) for res in nonconv_counts]
mins = [min(res) for res in nonconv_counts]
maxes = [max(res) for res in nonconv_counts]

# Plot mean, min, max
plt.plot(stop_points, [0 for sp in stop_points], color="blue", linestyle="--")
plt.fill_between(stop_points, mins, maxes, color="grey", label="Range over {} replicates".format(n_replicates))
plt.plot(stop_points, means, color="k", label="Mean over {} replicates".format(n_replicates))

plt.legend()

plt.xlim(stop_points[0],stop_points[-1])
plt.xticks(stop_points, rotation=45.0)

plt.ylabel("Non-converged edge probabilities", family="serif")
plt.xlabel("Sampling Chain Length", family="serif")

plt.title("MCMC Convergence: {} variables".format(v), family="serif")
plt.tight_layout()
#savefig
plt.savefig(snakemake.output[0])

