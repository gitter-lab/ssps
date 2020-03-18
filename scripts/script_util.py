# script_util.py
# 
# Useful functions for scripting within Snakemake

import json
import sys
import os
import re
import pandas as pd


NICE_NAMES = {"mcmc_d=1": "MCMC",
              "hill": "Hill's Exact",
              "lasso": "LASSO",
              "funchisq": "FunChisq",
              "prior_baseline": "Prior knowledge (baseline)"
              "v": "$V$",
              "r": "$r$",
              "a": "$a$",
              "t": "$t$",
              "aucroc_mean": "Mean AUCROC",
              "aucpr_mean": "Mean AUCPR"
              "psrf": "PSRF",
              "neff": "$N_{\text{eff}}$"
              }


def parse_path_kvs(file_path):
    """
    Find all key-value pairs in a file path;
    the pattern is *_KEY=VALUE_*.
    """
    parser = re.compile("(?<=[/_])[a-z0-9]+=[a-zA-Z0-9]+[.]?[0-9]*(?=[_/.])")

    kvs = parser.findall(file_path)
    kvs = [kv.split("=") for kv in kvs]
    
    return {kv[0]: to_number(kv[1]) for kv in kvs}


def to_number(num_str):
    try:
        return int(num_str)
    except ValueError:
        try:
            return float(num_str) 
        except ValueError:
            return num_str


def tabulate_results(filenames, key_lists, map_fn=None):
    """
    General purpose function for extracting 
    values stored in JSON files and building
    a DataFrame from them. 

    Receive a list of filenames and a list
    of key *lists*. For each file, create a row
    in the dataframe containing the values
    located by the key_lists. 

    Optionally: apply a function `map_fn` to the
    data before tabulating it.
    """
    if map_fn is None:
        map_fn = lambda x: x

    table = pd.DataFrame()
    
    for fname in filenames:
        
        f = open(fname, "r")
        file_data = json.load(f)
        f.close()

        row = parse_path_kvs(fname)
        for kl in key_lists:
            v = file_data
            for k in kl:
                v = v[k]
            row["_".join(kl)] = map_fn(v)

        table = table.append(row, ignore_index=True)

    return table
   

def extract_from_file(filename, key_list=[]):
    """
    Assume `filename` is the path to a json file.
    Extract the data located by `key_list`.
    """

    f = open(filename,"r")
    file_data = json.load(f)
    f.close()
    for k in key_list:
        file_data = file_data[k]
    return file_data


