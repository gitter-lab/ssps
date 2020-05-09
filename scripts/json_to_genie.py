import json
import pandas as pd
import sys

"""
Converts lists of weighted parent sets into 
a dataframe of weighted edges
"""
def psets_to_edgedf(parent_sets, node_names=None):

    if node_names is None:
        node_names = ["node_{}".format(i) for i in range(len(parent_sets))]

    edge_df = pd.DataFrame()

    for j, ps in enumerate(parent_sets):
        for i, p_prob in enumerate(ps):
            row = {0: node_names[i], 
                   1: node_names[j], 
                   2: p_prob}
            edge_df = edge_df.append(row, ignore_index=True)
    
    edge_df.sort_values(2, ascending=False, inplace=True)

    return edge_df


if __name__=="__main__":

    # get arguments
    pred_file = sys.argv[1]
    out_file = sys.argv[2]
    node_names = None

    # optionally: read in a JSON file of node names
    if len(sys.argv) > 3:
        name_file = sys.argv[3]
        with open(name_file, "r") as f:
            node_names = json.load(f)

    # load prediction file
    with open(pred_file) as f:
        preds = json.load(f)

    # convert parent sets 
    # to edge-list dataframe
    parent_sets = preds[preds["edge_conf_key"]]
    edge_df = psets_to_edgedf(parent_sets, node_names=node_names)

    # output GENIE3
    edge_df.to_csv(out_file, sep="\t", header=False, index=False)


