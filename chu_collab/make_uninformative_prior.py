
import pandas as pd
import numpy as np
import json
import sys

if __name__=="__main__":

    gene_name_file = sys.argv[1]
    outfile = sys.argv[2]

    gene_names = json.load(open(gene_name_file, "r"))
    
    n = len(gene_names)

    values = np.ones((n,n)) * (5.0 / n)

    df = pd.DataFrame(data=values)

    df.to_csv(outfile, sep=",", index=False, header=False)
