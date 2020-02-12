# preprocess_dream.py
# David Merrell
# 2020-02
# 
# Preprocess DREAM8 HPN data for MCMC/FunChisq/Hill/etc.


import pandas as pd
import os
import argparse


def to_minutes(timestr):
    """
    Convert a time string (e.g., '10min') to a floating point
    number of minutes (e.g., 60.0)
    """
    
    if timestr[-3:] == "min":
        num = float(timestr[:-3])
    elif timestr[-2:] == "hr":
        num = float(timestr[:-2]) * 60.0

    return num


def load_dream_ts(csv_path, keep_start=False):
    """
    Read in a DREAM challenge time series CSV file
    and return a DataFrame with appropriate columns.
    """

    # the original CSV is strangely formatted -- it has
    # an extra column and a multi-line header.
    df = pd.read_csv(csv_path)
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.iloc[2,3:] = df.iloc[1,3:].values
    df.columns = df.loc[2,:].values
    df = df.loc[3:,:]
    df.index = range(df.shape[0])


    # The original format doesn't give a "Stimulus" label
    # at timepoint 0; we'll restore the label if necessary
    if keep_start:
        print("TOO BAD.") 
        df = df[df["Stimulus"].isnull() == False]
    else:
        # Otherwise, just remove these rows.
        df = df[df["Stimulus"].isnull() == False]

    return df


def create_standard_dataframe(dream_df, ignore_stim=False,
                                         ignore_inhib=False):
    """
    For each context contained in `dream_df`, create a time series
    dataframe.
    """
    
    context_cols = []
    if not ignore_inhib:
        context_cols.append("Inhibitor")
    if not ignore_stim:
        context_cols.append("Stimulus")

    joiner = lambda x: "_".join(x)

    dream_df["context"] = df[context_cols].apply(joiner, axis=1) 
    contexts = dream_df["context"].unique()
    
    dream_df.rename(columns={"Timepoint": "timestep"}, inplace=True)
    dream_df.loc[:,"timeseries"] = dream_df[["Inhibitor","Stimulus"]].apply(joiner, axis=1)
   
    dream_df.sort_values(["context","timeseries","timestep"], inplace=True)

    keep_cols = ["context", "timeseries", "timestep"]

    # IMPORTANT: standard order of variables = lexicographic
    var_cols = sorted([c for c in dream_df.columns if c not in keep_cols+["Inhibitor", "Stimulus"]])

    dream_df = dream_df[keep_cols + var_cols]
    
    return dream_df 


if __name__=="__main__":

    # Get command line args
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("timeseries_file", help="path to a DREAM challenge time series CSV file")
    parser.add_argument("output_dir", help="directory where the output CSVs will be written")
    parser.add_argument("--ignore-stim", help="Do NOT treat different stimuli as different contexts.",
                        action="store_true")
    parser.add_argument("--ignore-inhibitor", help="Do NOT treat different inhibitors as different contexts.", 
                        action="store_true")
    parser.add_argument("--keep-start", help="Keep the time series data at timepoint 0",
                        action="store_true")
    args = parser.parse_args()
   
    ts_filename = str(args.timeseries_file)
    ignore_stim = args.ignore_stim
    ignore_inhib = args.ignore_inhibitor

    # Load the DREAM challenge data
    df = load_dream_ts(ts_filename, keep_start=args.keep_start)

    # transform these columns into more useful forms
    df["Timepoint"] = df["Timepoint"].map(to_minutes)
    df.loc[df["Inhibitor"].isnull(), "Inhibitor"] = "nothing"
    df.loc[df["Stimulus"].isnull(), "Stimulus"] = "nothing"
    
    # Convert the data to (context-specific) time series dataframes,
    # formatted correctly for our analysis
    new_ts_df = create_standard_dataframe(df, ignore_stim=ignore_stim, 
                                              ignore_inhib=ignore_inhib)

    in_fname = os.path.basename(ts_filename)
    cell_line = in_fname.split("_")[0] 

    contexts = new_ts_df["context"].unique()
    
    for ctxt in contexts:
        ctxt_str = "{}_{}".format(cell_line, ctxt)
        out_df = new_ts_df[new_ts_df["context"] == ctxt]
        out_df.iloc[:,1:].to_csv(os.path.join(str(args.output_dir), ctxt_str+".csv"),
                                 sep="\t", index=False)


 
