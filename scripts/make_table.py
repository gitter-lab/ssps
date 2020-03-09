import script_util as su
import pandas as pd
import sys
import numpy as np

input_files = sys.argv[1:-1]
output_file = sys.argv[-1]

AUCPR_STR = "aucpr"
AUCROC_STR = "aucroc"

keys = ["cl", "stim"]

table = su.tabulate_results(input_files, [[AUCPR_STR],[AUCROC_STR]])

#table["n_replicates"] = table[AUCPR_STR].map(len)
table[AUCROC_STR+"_mean"] = table[AUCROC_STR].map(np.mean)
table[AUCROC_STR+"_std"] = table[AUCROC_STR].map(np.std)
table[AUCPR_STR+"_mean"] = table[AUCPR_STR].map(np.mean)
table[AUCPR_STR+"_std"] = table[AUCPR_STR].map(np.std)

table.drop([AUCPR_STR, AUCROC_STR], axis=1, inplace=True)

methods = [f.split("/")[-2] for f in input_files]
table["method"] = methods
keys.append("method")

#table = table[["v", "r", "a", "method", AUCPR_STR+"_mean", AUCPR_STR+"_std", AUCROC_STR+"_mean", AUCROC_STR+"_std", "n_replicates"]]
table = table[keys+[AUCPR_STR+"_mean", AUCPR_STR+"_std", AUCROC_STR+"_mean", AUCROC_STR+"_std"]]
table.sort_values(keys, inplace=True)

table.to_csv(output_file, index=False, sep="\t")
