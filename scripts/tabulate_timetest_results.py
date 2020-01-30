# tabulate_timetest_results.py
# 2020-01
# David Merrell
#
# Makes a table of execution times for the 
# Hill et al. 2012 Exact DBN method.

import json
import sys
import os
import pandas as pd
import script_util as su

HILL_TIME_PARAMS = snakemake.config["hill_timetest"]
HILL_MODES = HILL_TIME_PARAMS["modes"]
degs = [str(d) for d in HILL_TIME_PARAMS["degs"]]
vs = [str(v) for v in HILL_TIME_PARAMS["v"]]

def get_times(fnames):

    times = pd.DataFrame(index=pd.MultiIndex.from_product([HILL_MODES, degs]), columns=vs)
    for fname in fnames:

        kvs = su.parse_path_kvs(fname)

        res_dict = json.load(open(fname, "r"))
        v = kvs["v"]
        split_dirname = fname.split(os.path.sep)[-2].split("_")
        deg = kvs["deg"]
        mode = kvs["mode"]

        if res_dict['timed_out']:
            times.loc[(mode,deg),v] = "timeout"
        else:
            times.loc[(mode,deg),v] = str(res_dict['time'])
 
    return times


time_df = get_times(snakemake.input)
time_df.to_csv(snakemake.output[0])


