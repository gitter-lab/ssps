
import pandas as pd
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True

from matplotlib import pyplot as plt
import sys
import os
import argparse
import script_util as su


def make_barchart(ax, relevant, method_names, score_str):

    gp = relevant.groupby("method")

    means = gp[score_str].mean()
    means = means[method_names] 

    mins = gp[score_str].min()
    mins = mins[method_names]
    eminus = means - mins

    maxes = gp[score_str].max()
    maxes = maxes[method_names]
    eplus = maxes - means

    n_methods = len(method_names)
    x = range(n_methods)

    colors=['grey','r','b','orange','c','y']
    our_colors = [colors[i%len(colors)] for i in range(n_methods)]

    bars = ax.bar(x, means, color=our_colors, edgecolor='k',
                            yerr=[eminus,eplus], capsize=5.0)

    ax.set_xticks(x)
    ax.set_xticklabels([su.SHORT_NAMES[m] for m in method_names], rotation='vertical')
    ax.set_yticks([0,1])

    return bars


if __name__=="__main__":

    args = sys.argv
    score_file = args[1]
    out_file = args[2]
    score_str = args[3]
    method_names = args[4:]

    df = pd.read_csv(score_file, sep="\t")
    
    stims = df["stim"].unique()
    cells = df["cl"].unique()

    df.set_index(["cl","stim"], inplace=True)

    n_rows = len(cells)
    n_cols = len(stims)

    fig, axarr = plt.subplots(n_rows, n_cols, 
                              sharey=True, sharex=True, 
                              figsize=(n_cols*1.5,n_rows*1.5 + 0.5))

    for i, cl in enumerate(cells):
        for j, stim in enumerate(stims):

            ax = axarr[i][j]

            print("\t",cl,stim)
            relevant = df.loc[(cl,stim),:]
            print(relevant)

            #in_method_names = lambda x: x in method_names
            #relevant_scores = df.loc[df["method"].map(in_method_names) , score_col]

            bars = make_barchart(ax, relevant, method_names, score_str)

            if i == len(cells) - 1:
                ax.set_xlabel(stim)
            if j == 0:
                ax.set_ylabel(cl)

    plt.suptitle("HPN-DREAM Challenge: {}".format(su.NICE_NAMES[score_str]))
    plt.tight_layout(rect=(0,0,1,0.95))
    plt.savefig(out_file, dpi=300)


