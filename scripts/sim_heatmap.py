# "sim_heatmap.py"
# David Merrell
# 
# Plot heatmaps displaying simulation study results.

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



def compute_t_statistics(df, test_key_cols, sample_col, qty_cols,
                             method_col, baseline_name):

    methods = df[method_col].unique().tolist()
    baseline = df[df[method_col] == baseline_name]
    
    key_arrs = [df[k].unique() for k in test_key_cols+[method_col]]
    
    df.set_index(test_key_cols+[method_col], inplace=True)
    baseline.set_index(test_key_cols, inplace=True)
    
    result_df = pd.DataFrame(index=pd.MultiIndex.from_product(key_arrs),
                             columns=qty_cols)

    for ks in result_df.index:
   
        diffs = df.loc[ks, qty_cols] - baseline.loc[ks[:-1], qty_cols].values
        diffs.reset_index(inplace=True)
                        
        gp = diffs.groupby(by=test_key_cols)
        means = gp[qty_cols].mean()
        stds = gp[qty_cols].std()
        n = means.shape[0]
               
        f = lambda x: x / np.sqrt(n)
        ses = stds.apply(f)
        ts = means / ses

        result_df.loc[ks, qty_cols] = ts.values

    result_df.index.rename(test_key_cols+[method_col], inplace=True)
    result_df.reset_index(inplace=True)
    return result_df


def aggregate_scores(table, key_cols, score_cols):

    gp = table.groupby(key_cols)
    agg = gp[score_cols].mean()
    agg.reset_index(inplace=True)

    return agg 


def make_heatmap(ax, relevant, x_col, y_col, qty_col, **kwargs):

    x_vals = relevant[x_col].unique()
    y_vals = relevant[y_col].unique()

    grid = np.zeros((len(y_vals), len(x_vals)))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            grid[j,i] = relevant.loc[(relevant[x_col] == x) & (relevant[y_col] == y), qty_col]

    img = ax.imshow(grid, origin="lower", **kwargs)
    ax.set_xticks(list(range(len(x_vals))))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(list(range(len(y_vals))))
    ax.set_yticklabels(y_vals)
    #ax.set_xlim([-0.5, len(x_vals)-0.5])
    #ax.set_ylim([-0.5, len(y_vals)-0.5])

    #ax.label_outer()
    return img


def subplot_heatmaps(qty_df, macro_x_col, macro_y_col, 
                     micro_x_col, micro_y_col, qty_col, score_str,
                     output_filename="simulation_scores.png",
                     macro_x_vals=None, macro_y_vals=None,
                     cmap="Greys", vmin=None, vmax=None):

    if macro_x_vals is None:
        macro_x_vals = qty_df[macro_x_col].unique().tolist()

    if macro_y_vals is None:
        macro_y_vals = qty_df[macro_y_col].unique().tolist()

    n_rows = len(macro_y_vals)
    n_cols = len(macro_x_vals)

    fig, axarr = plt.subplots(n_rows, n_cols, 
                              sharey=True, sharex=True, 
                              figsize=(2.0*n_cols,2.0*n_rows))
  

    in_macro_y_vals = lambda x: x in macro_y_vals
    relevant_scores = qty_df.loc[qty_df[macro_y_col].map(in_macro_y_vals) , qty_col]
    
    if vmin is None:
        vmin = relevant_scores.quantile(0.05)
    if vmax is None:
        vmax = relevant_scores.quantile(0.95)

    nrm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=nrm, cmap=cmap)

    imgs = []

    # Iterate through the different subplots
    for i, myv in enumerate(macro_y_vals):
        for j, psize in enumerate(macro_x_vals):
            
            ax = axarr[i][j]
            relevant = qty_df.loc[(qty_df[macro_y_col] == myv) & (qty_df[macro_x_col] == psize),:]

            img = make_heatmap(ax, relevant, micro_x_col, micro_y_col, qty_col, norm=nrm, cmap=cmap)
            imgs.append(img)
           
            #ax.set_xlim([0,3])
            #ax.set_ylim([0,3])
            if i == len(macro_y_vals)-1:
                ax.set_xlabel("${}$\n$V$ = {:d}".format(micro_x_col, int(psize)),family='serif')
            if j == 0:
                ax.set_ylabel("{}\n${}$".format(su.NICE_NAMES[myv], micro_y_col),family='serif')
    
    
    fig.suptitle("Simulation Study: {}".format(su.NICE_NAMES[score_str]),family='serif',fontsize=16)
    plt.tight_layout(rect=[0.0,0.0,1,0.95])
    fig.colorbar(imgs[-1], ax=axarr, location="top", shrink=0.8, pad=0.05, fraction=0.05, use_gridspec=True)

    plt.savefig(output_filename, dpi=300)#, bbox_inches="tight")


if __name__=="__main__":

    args = sys.argv
    infile = args[1]
    mean_outfile = args[2]
    t_outfile = args[3]
    score_str = args[4]
    baseline_name = args[5]
    methods = args[6:]
    
    table = pd.read_csv(infile, sep="\t") 

    key_cols = ["v","r","a"]
    sample_col = "replicate"
    score_cols = [score_str]
    method_col = "method"
  
    aggregate_table = aggregate_scores(table, key_cols + [method_col], score_cols)
    #aggregate_table.to_csv("means.tsv", sep="\t")
    
    t_stat_table = compute_t_statistics(table, key_cols, sample_col, score_cols,
                                        method_col, baseline_name)
    #t_stat_table.to_csv("t_statistics.tsv", sep="\t")

    print(mean_outfile)
    subplot_heatmaps(aggregate_table, "v", "method", "r", "a", score_str, score_str,
                     output_filename=mean_outfile, macro_y_vals=methods+[baseline_name],
                     cmap="Greys")

    print(t_outfile)
    subplot_heatmaps(t_stat_table, "v", "method", "r", "a", score_str, "t_stat_{}".format(score_str), 
                     output_filename=t_outfile, macro_y_vals=methods,
                     cmap="RdBu", vmin=-5.0, vmax=5.0) 


