
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


def make_barchart(ax, relevant, score_str):

    gp = relevant.groupby("method")

    means = gp[score_str].mean()
    mins = gp[score_str].min()
    maxes = gp[score_str].max()

    n_methods = len(relevant["method"].unique())

    bars = ax.bar(range(n_methods), means)

    return bars


if __name__=="__main__":

    df = pd.read_csv("dream_scores.tsv", sep="\t")
    
    stims = df["stim"].unique()
    cells = df["cl"].unique()

    df.set_index(["cl","stim"], inplace=True)

    n_rows = len(cells)
    n_cols = len(stims)

    fig, axarr = plt.subplots(n_rows, n_cols, 
                              sharey=True, sharex=True, 
                              figsize=(n_cols*2.54,n_rows*2.54))

    for i, cl in enumerate(cells):
        for j, stim in enumerate(stims):

            print("\t",cl,stim)
            relevant = df.loc[(cl,stim),:]
            print(relevant)
            bars = make_barchart(axarr[i][j], relevant, "aucpr")

    plt.savefig("dumb_barchart.png", dpi=300)
