# convergence_viz.py
#
# Visualize convergence analysis results.

import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import script_util as su

print("I'M THE INPUT: ", snakemake.input)
print("I'M THE OUTPUT: ", snakemake.output[0])

#table = su.tabulate_results(snakemake.input, )
#table.to_csv(snakemake.output[0], index=False)

f = open(snakemake.output[0], "w")
f.write("stuff")
f.close()
