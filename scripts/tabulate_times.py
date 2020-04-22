import script_util as su
import pandas as pd
import sys
import numpy as np
import os


if __name__=="__main__":

    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]
   
    TIME_STR = "t_elapsed" 
    N_STR = "n"
    
    table = su.tabulate_results(input_files, [[TIME_STR], [N_STR]], 
                                verbose=True, map_fn=sum)
    
    table.to_csv(output_file, index=False, sep="\t")


