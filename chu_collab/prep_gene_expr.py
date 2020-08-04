import pandas as pd
import sys
import json

def prep_chu_timeseries(in_df):

    in_df.set_index("gene_id", inplace=True)
    in_df = in_df.iloc[:,:-5]
    
    col_map_fn = lambda x: int(x.split("_")[1])
    in_df.columns = in_df.columns.map(col_map_fn)

    out_df = in_df.transpose()
    out_df.index.rename("timestep", inplace=True)
    out_df["timeseries"] = 1.0
    out_df.reset_index(inplace=True)
    out_df = out_df[["timeseries"] + out_df.columns[:-1].tolist()]
    
    return out_df


if __name__=="__main__":

    infile = sys.argv[1]
    ts_outfile = sys.argv[2]
    gene_names_outfile = sys.argv[3]

    in_df = pd.read_csv(infile, sep="\t")

    out_df = prep_chu_timeseries(in_df)

    out_df.to_csv(ts_outfile, sep="\t", index=False)

    genes = out_df.columns.tolist()[2:]
    json.dump(genes, open(gene_names_outfile, "w"))


