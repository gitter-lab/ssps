# sim_study_score_viz.py
#
# Visualize a method's scores on simulated datasets.

import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import script_util as su

print("I'M THE INPUT: ", snakemake.input)
print("I'M THE OUTPUT: ", snakemake.output[0])


table = su.tabulate_results(snakemake.input, [["auprc"],["auroc"]])

pr = su.extract_from_file(snakemake.input[1], ["pr_curves",0])

plt.plot(pr[0],pr[1])
plt.xlim(0.0,1.0)
plt.ylim(0.0,1.0)
plt.savefig(snakemake.output[0])

#table.to_csv(snakemake.output[0], index=False)
