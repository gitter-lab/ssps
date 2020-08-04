"""
concatenate_samples.py
(c) David Merrell 2020-08-03

Concatenate the samples contained in multiple JSON files,
yielding a single combined sample file.
"""

import json
import sys


def concatenate_all_samples(filenames):

    with open(filenames[0], "r") as f:
        result = json.load(f)

    for fname in filenames[1:]:
        with open(fname, "r") as fnew:
            newsamples = json.load(fnew)
        result = concatenate_samples(result, newsamples)

    return result


def concatenate_samples(samples1, samples2):

    samples1["t_elapsed"] += samples2["t_elapsed"]

    # Concatenate lambda samples
    for i, _ in enumerate(samples1["lambda"]):
        
        v = samples2["lambda"][i][1:]
        
        # Need to update the sample index for the 
        # concatenated samples
        for j, pair in enumerate(v):
            v[j][0] += samples1["n"]
        
        samples1["lambda"][i] += v 

    # Concatenate parent set samples
    for i, pdict in enumerate(samples2["parent_sets"]):
        for p, v in pdict.items():
            if v[0][0] == 0:
                v = v[1:]

            # Need to update the sample index for the 
            # concatenated samples
            for j, pair in enumerate(v):
                v[j][0] += samples1["n"]

            if p not in samples1["parent_sets"][i]:
                samples1["parent_sets"][i][p] = []

            samples1["parent_sets"][i][p] += v

    # add n's
    samples1["n"] += samples2["n"]
    
    return samples1


if __name__=="__main__":


    sample_files = sys.argv[1:-1]
    out_file = sys.argv[-1]

    result = concatenate_all_samples(sample_files)

    with open(out_file, "w") as f:
        json.dump(result, f)

