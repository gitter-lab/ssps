# score_dream.py
#
# Script for scoring dream submissions.
#
# DREAMtools.py was more trouble than it was worth -- required
# old versions of python, hampering our reproducibility efforts
# (i.e., Snakemake)

import numpy as np
import pandas as pd
import argparse
import json
import os

def load_predictions(pred_filename):

    d = json.load(open(pred_filename))
    edge_conf_key = d["edge_conf_key"]

    return d[edge_conf_key]


def load_true_desc(desc_filename):

    df = pd.read_csv(desc_filename, sep=",", header=None)
    tru = df.loc[0,:]

    # exclude any "bad" antibodies
    tru = tru[pd.isnull(tru) == False]
    descendants = set([i for i, v in enumerate(tru) if v == 1])
        
    return descendants 



def compute_scores(pred_file, desc_file, antibody_file, root_antibody):
  
    with open(antibody_file, "r") as abf: 
        antibody_idx = json.load(abf)[root_antibody] 
    
    # (probabilistic) parent sets 
    parent_pred = load_predictions(pred_file)
    # "ground truth" descendant set 
    true_desc = load_true_desc(desc_file) 

    score_tuple = descendant_set_auc(parent_pred, true_desc, antibody_idx, return_curves=True)

    score_dict = {"aucpr": score_tuple[0],
                  "aucroc": score_tuple[1],
                  "pr_curves": score_tuple[2],
                  "roc_curves": score_tuple[3]
                 }

    return score_dict


def scoring_dfs(vertex, edge_list, reached):
    reached.add(vertex)
    for child in edge_list[vertex].copy():
        edge_list[vertex].remove(child)
        if child not in reached:
            scoring_dfs(child, edge_list, reached)
    
    return edge_list, reached


def fp_tp(pred_desc, true_desc):
    tp = len(pred_desc.intersection(true_desc))
    fp = len(pred_desc.difference(true_desc))
    return fp, tp


def prec_rec(fp, tp, npos):
    if fp+tp == 0:
        prec = 1.0
    else:
        prec = tp / (fp+tp)
    return prec, tp/npos


def fpr_tpr(fp, tp, nneg, npos):
    return fp/nneg, tp/npos
    
    
def fpr_tpr_prec_rec(pred_desc, true_desc, nneg, npos):
    fp, tp = fp_tp(pred_desc, true_desc)
    fpr, tpr = fpr_tpr(fp, tp, nneg, npos) 
    prec, rec = prec_rec(fp, tp, npos)
    
    return fpr, tpr, prec, rec


def increment_trap(auc, x, y, old_x, old_y):
    return auc + 0.5*(y + old_y)*(x - old_x)

def increment_rect(auc, x, y, old_x, old_y):
    return auc + y*(x - old_x)

def descendant_set_auc(prob_parents, true_desc, root_idx, return_curves=True):
    
    V = len(prob_parents)
    npos = len(true_desc)
    nneg = V - npos
    
    # These things manage the state of 
    # the descendant algorithm
    descendants = set([])
    edges = [(i, j, prob) for (j, ps) in enumerate(prob_parents) for (i, prob) in enumerate(ps)]
    edges = sorted(edges, key=lambda x: x[2])
    unused_edges = [[] for i in range(V)]
    prev_prob = np.inf
    
    # score-related quantities
    old_fpr = 0.0
    old_tpr = 0.0
    old_prec = 1.0
    old_rec = 0.0
    roc_curve = [[old_fpr],[old_tpr]]
    pr_curve = [[old_rec],[old_prec]]
    auc_roc = 0.0
    auc_pr = 0.0
    
    i = 0
    while len(edges) > 0:
        parent, child, prob = edges.pop()
        
        if (child not in descendants):
            if (parent == root_idx) or (parent in descendants):
                #unused_edges[parent].append(child)
                unused_edges, descendants = scoring_dfs(child, unused_edges, descendants)
            # This edge may be useful in the future.
            else:
                unused_edges[parent].append(child)
            
        if prob < prev_prob and i > 0:
            # UPDATES!
            
            fpr, tpr, prec, rec = fpr_tpr_prec_rec(descendants, true_desc, nneg, npos)
                
            auc_roc = increment_trap(auc_roc, fpr, tpr, old_fpr, old_tpr)
            auc_pr = increment_rect(auc_pr, rec, prec, old_rec, old_prec)
            old_fpr, old_tpr, old_prec, old_rec = fpr, tpr, prec, rec
            roc_curve[0].append(fpr)
            roc_curve[1].append(tpr)
            pr_curve[0].append(rec)
            pr_curve[1].append(prec)

        prev_prob = prob 
        i += 1
    
    # UPDATES!
    fpr, tpr, prec, rec = fpr_tpr_prec_rec(descendants, true_desc, nneg, npos)
    auc_roc = increment_trap(auc_roc, fpr, tpr, old_fpr, old_tpr)
    auc_pr = increment_rect(auc_pr, rec, prec, old_rec, old_prec)
    roc_curve[0].append(fpr)
    roc_curve[1].append(tpr)
    pr_curve[0].append(rec)
    pr_curve[1].append(prec)
    
    return auc_roc, auc_pr, roc_curve, pr_curve
        


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("pred_file", help="path to JSON file containing edge predictions")
    parser.add_argument("true_desc_file", help="path to DREAM challenge descendant-set CSV file")
    parser.add_argument("antibody_file", help="path to JSON file containing the indices of antibodies")
    parser.add_argument("output_file", help="path to an output JSON file")
    parser.add_argument("--root-antibody", help="Name of the root antibody for our descendant sets",
                                           default="mTOR_pS2448")
    args = parser.parse_args()

    score_dict = compute_scores(args.pred_file, args.true_desc_file, 
                                args.antibody_file, args.root_antibody)

    with open(args.output_file, "w") as f:
        json.dump(score_dict, f)

    
