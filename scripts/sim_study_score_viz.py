# sim_study_score_viz.py
#
# Visualize a method's scores on simulated datasets.

import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import script_util as su
import numpy as np

print("I'M THE INPUT: ", snakemake.input)
print("I'M THE OUTPUT: ", snakemake.output[0])

fig_dir = os.path.dirname(snakemake.output[0])
simfig_dir = os.path.join(fig_dir, "simulation_study")
scores_dir = os.path.join(simfig_dir, "abs_scores")
compare_dir = os.path.join(simfig_dir, "rel_scores")
pr_curve_dir = os.path.join(simfig_dir, "pr_curves")

table = su.tabulate_results(snakemake.input, [["auprc"],["auroc"]])
table.to_csv(snakemake.output[0], index=False)

method_str = snakemake.wildcards[0]
score_str = "mean_aucpr"
my_mean = lambda x: sum(x)/len(x)

# Average scores over replicates
table[score_str] = table["auprc"].map(my_mean)
table["mean_aucroc"] = table["auroc"].map(my_mean) 

# Get the unique combinations of (problem size) x (mcmc_config)
vtd_combs = table[["v","t","d"]]
table["vtd_combs"] = vtd_combs.apply(lambda x: tuple(x), axis=1)
u_vtd = table["vtd_combs"].unique()

# for each combination, produce a score heatmap
for comb in u_vtd:

    relevant = table[table["vtd_combs"] == comb]
    u_r = table["r"].unique()
    u_r = sorted(u_r, reverse=True)
    u_a = table["a"].unique()

    result_mat = np.zeros((len(u_r),len(u_a)))

    means = relevant.groupby(["r","a"])[score_str].mean()
    for i, r in enumerate(u_r):
        for j, a in enumerate(u_a):
            result_mat[i,j] = means[(r,a)]

    plt.imshow(result_mat, cmap="Greys")#, vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(range(len(u_a)), u_a)
    plt.yticks(range(len(u_r)), u_r)
    plt.xlabel("a")
    plt.ylabel("r")
    plt.title("{}: {}".format(score_str, comb))
    comb_str = "_".join(["{}".format(c) for c in comb])
    plt.savefig(os.path.join(scores_dir, "{}_{}.png".format(score_str, comb_str)))



 
