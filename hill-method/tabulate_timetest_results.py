# tabulate_timetest_results.py
# 2020-01
# David Merrell
#
# Makes a table of execution times for the 
# Hill et al. 2012 Exact DBN method.

import matplotlib.pyplot as plt
import json
import sys
import os
import pandas as pd


def get_times(fnames):

    times = pd.DataFrame(index=pd.MultiIndex.from_product([snakemake.HILL_MODES,
                                                           snakemake.HILL_TIME_PARAMS["degs"]), 
                         columns=snakemake.HILL_TIME_PARAMS["v"])
    for fname in fnames:
        res_dict = json.load(open(fname, "r"))
        v = os.path.basename(fname).split("_")[0].split("=")[1]
        split_dirname = fname.split(os.path.sep)[-2].split("_")
        deg = split_dirname[1]
        mode = split_dirname[2]

        times.loc[(mode,deg), v] = res_dict['time']
 
    return times


time_df = get_times(snakemake.input)
time_df.to_csv(snakemake.output, header=False)


