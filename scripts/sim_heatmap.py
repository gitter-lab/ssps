# "sim_heatmap.py"
# David Merrell
# 
# Make a plot of simulation study results.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import argparse
import script_util as su


def make_heatmap(ax, relevant):

    u_r = table["r"].unique()
    u_r = sorted(u_r, reverse=True)
    u_a = table["a"].unique()

    result_mat = np.zeros((len(u_r),len(u_a)))

    means = relevant.groupby(["r","a"])[score_str].mean()
    for i, r in enumerate(u_r):
        for j, a in enumerate(u_a):
            result_mat[i,j] = means[(r,a)]

    ax.imshow(result_mat, cmap="Greys")#, vmin=0.0, vmax=1.0)

    return ax


def compute_t_statistics(df, test_key_cols, sample_col, qty_cols,
                             method_col, baseline_name):

    # test_key_cols = ["v","r","a"]
    # sample_col = "replicate"
    # qty_cols = ["aucroc", "aucpr"]
    # method_col = "method"
    # baseline_name = "prior_baseline"

    methods = df[method_col].unique().tolist()
    baseline = df[df[method_col] == baseline_name]
    
    df.set_index(test_key_cols + [sample_col], inplace=True)
    baseline.set_index(test_key_cols + [sample_col], inplace=True)
    
    diff_cols = ["{}_diff".format(c) for c in qty_cols]
    
    result_df = pd.DataFrame()
    
    for method in methods:
    
        temp = pd.DataFrame()
        temp[diff_cols] = df.loc[df[method_col] == method, qty_cols] - baseline[qty_cols]
        temp.reset_index(inplace=True)
                        
        gp = temp.groupby(by=test_key_cols)
        means = gp[diff_cols].mean()
        result_df[["{}_{}_mean".format(method, qty) for qty in qty_cols]] = means
        stds = gp[diff_cols].std()
                    
        result_df[["{}_{}_std".format(method, qty) for qty in qty_cols]] = stds
        ns = gp[sample_col].count()
        result_df["{}_n".format(method)] = ns
               
        f = lambda x: x / ns.map(np.sqrt)
        ses = stds.apply(f)
        result_df[["{}_{}_se".format(method, qty) for qty in qty_cols]] = ses
        ts = means / ses
        result_df[["{}_{}_t".format(method, qty) for qty in qty_cols]] = ts
                  
    return result_df


def subplot_heatmaps(score_df, method_keys, baseline_key,
                     output_filename="simulation_scores.png"):

    problem_sizes = score_df["v"].unique().tolist()

    fig, axarr = plt.subplots(len(problem_sizes), len(method_keys), sharey=True, sharex=True, figsize=(3.0*2.54,3.0*2.54),dpi=300)
    print(axarr)
   
    # Iterate through the different subplots
    for i, method in enumerate(method_keys):
        for j, psize in enumerate(problem_sizes):
            # count rows from the bottom, count cols ltr.
            row = len(method_keys) - i - 1
            ax = axarr[j][row]

            relevant = score_df.loc[(score_df["method"] == method) & (score_df["v"] == psize),:]
            #relevant = relevant.groupby("replicate").mean().reset_index(inplace=True)

            ax = make_heatmap(ax, relevant)
            #ax.set_xlim([0,3])
            #ax.set_ylim([0,3])
            #ax.set_xticks([0,1,2,3], )
            #ax.set_yticks([0,1,2,3], )
            if row == len(methods)-1:
                ax.set_xlabel("{}".format(su.NICE_NAMES[method]),family='serif')
            if j == 0:
                ax.set_ylabel("$V$ = {}".format(psize),family='serif')
    
    fig.text(0.02,0.8, "Method",ha='center',family='serif',rotation="vertical",fontsize=14) 
    fig.text(0.5,0.02, "Problem size",ha='center',family='serif',fontsize=14) 
    #plt.tight_layout(rect=[0.03,0.03,1,0.95])
    fig.suptitle("Simulation Study Performance",family='serif',fontsize=16)

    plt.savefig(output_filename)


if __name__=="__main__":

    args = sys.argv
    infile = args[1]
    outfile = args[2]
    score_str = args[3]
 
    methods = args[4:]

    table = pd.read_csv(infile, sep="\t") 
    subplot_heatmaps(table, methods, "prior_baseline",
                     output_filename=outfile) 


