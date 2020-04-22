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


def get_times(filenames):

    df = su.tabulate_results(filenames, [["time"], ["timed_out"]], verbose=True)
    return df 


if __name__ == "__main__":

    infiles = sys.argv[1:-1]
    outfile = sys.argv[-1]
    
    time_df = get_times(infiles)
    time_df.to_csv(outfile, index=False, sep="\t")


