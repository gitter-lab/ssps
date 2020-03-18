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

def subplot_heatmaps(score_df, method_keys, baseline_key,
                     output_filename="simulation_scores.png"):

    problem_sizes = score_df["v"].unique().tolist()

    fig, axarr = plt.subplots(len(problem_sizes), len(method_keys), sharey=True, sharex=True, figsize=(3.0*2.54,3.0*2.54),dpi=300)
    axarr = [[axarr]]
   
    # Iterate through the different subplots
    for i, method in enumerate(method_keys):
        for j, psize in enumerate(problem_sizes):
            # count rows from the bottom, count cols ltr.
            row = len(method_keys) - i - 1
            ax = axarr[row][j]
            
            ax.scatter(np.array(id_dat['total_prob']), np.array(rot_dat['total_prob']),color='r',s=16)
            
            #ax.set_xlim([0,3])
            #ax.set_ylim([0,3])
            #ax.set_xticks([0,1,2,3], )
            #ax.set_yticks([0,1,2,3], )
            if row == len(dims)-1:
                ax.set_xlabel("{}".format(nice_names[conj]),family='serif')
            if j == 0:
                ax.set_ylabel("{} Dimensions".format(nice_names[dim]),family='serif')
    
    fig.text(0.02,0.8, "Method",ha='center',family='serif',rotation="vertical",fontsize=14) 
    fig.text(0.5,0.02, "Problem size",ha='center',family='serif',fontsize=14) 
    #plt.tight_layout(rect=[0.03,0.03,1,0.95])
    fig.suptitle("Simulation Study Performance",family='serif',fontsize=16)

    plt.savefig(output_filename)


if __name__=="__main__":

    args = sys.argv
    infiles = args[1:-1]
    outfile = args[-1]

    table = su.tabulate_results(infiles, [["aucpr"],["aucroc"]])


