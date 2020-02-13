# preprocess_dream_prior.py
#
#


import pandas as pd
import numpy as np
import argparse


def build_weighted_adj(eda_filename):

    df = pd.read_csv(eda_filename, sep=" ")
    df.reset_index(inplace=True)
    antibody_map = { a:i for (i,a) in enumerate(df["level_0"].unique()) }
    print(antibody_map)    

    V = len(antibody_map)
    adj = np.zeros((V,V))

    for (_, row) in df.iterrows():
        a = row["level_0"]
        b = row["level_2"]
        adj[antibody_map[a],antibody_map[b]] = row["EdgeScore"]

    print(adj)

    return adj


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("eda_file", help="path to a DREAM challenge time series CSV file")
    parser.add_argument("output_file", help="directory where the output CSVs will be written")
    args = parser.parse_args()

    adj_mat = build_weighted_adj(args.eda_file)

    df = pd.DataFrame(adj_mat)
    df.to_csv(args.output_file, sep=",", index=False, header=False) 

