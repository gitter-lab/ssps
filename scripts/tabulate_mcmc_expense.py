import script_util as su
import pandas as pd
import sys
import numpy as np
import os


def get_avg_parentset_neffs(parent_sets):

    total = 0.0
    n = 0
    for ps in parent_sets:
        for parent in ps:
            n += 1
            pair = parent[-1]
            
            if pair[0] is not None:
                total += pair[1]

    return total / n


if __name__=="__main__":

    out_file = sys.argv[1]
    time_file = sys.argv[2]
    pred_files = sys.argv[3:]
    
    rep_results = pd.read_csv(time_file, sep="\t")
    rep_results = rep_results.set_index(["v","r","a","replicate"])
    #rep_results = gp["n"].sum().to_frame()
    #rep_results["t_elapsed"] = gp["t_elapsed"].sum()
    print(rep_results)

    neff_table = su.tabulate_results(pred_files, [["conv_stats", "parent_sets"]], 
                                     map_fn=get_avg_parentset_neffs,
                                     verbose=True) 
    print(neff_table)
    neff_table = neff_table.set_index(["v","r","a","replicate"])
    rep_results["neff"] = neff_table["conv_stats_parent_sets"] 

    rep_results["n_per_hr"] = 3600.0 * rep_results["n"] / rep_results["t_elapsed"]
    rep_results["neff_per_hr"] = 3600.0 * rep_results["neff"] / rep_results["t_elapsed"] / 4.0

    rep_results.reset_index(inplace=True)

    print(rep_results)

    gp = rep_results.groupby(["v"])
    means = gp["n_per_hr"].mean().to_frame()
    means["neff_per_hr"] = gp["neff_per_hr"].mean()
    means.reset_index(inplace=True)
    means.to_csv(out_file, index=False, sep="\t")


