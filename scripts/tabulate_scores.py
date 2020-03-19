import script_util as su
import pandas as pd
import sys
import numpy as np
import os

if __name__=="__main__":

    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]
    
    AUCPR_STR = "aucpr"
    AUCROC_STR = "aucroc"
    
    table = su.tabulate_results(input_files, [[AUCPR_STR],[AUCROC_STR]])
    
    methods = [f.split(os.path.sep)[-2] for f in input_files]
    table["method"] = methods
    
    table.to_csv(output_file, index=False, sep="\t")


