
import json
from heapq import heapify, heappop
import sys
import matplotlib.pyplot as plt
import os
import script_util as su
import glob


def summarize_sequences(seq_dat):

    parent_seqs = seq_dat["parent_sets"]

    # Each vertex will have a heap containing 
    # "events" for its parent set
    parent_heaps = [] 
    for d in parent_seqs:
        h = [(p[0], p[1]) for ls in d.values() for p in ls]
        heapify(h)
        parent_heaps.append(h)

    results = []
    for heap in parent_heaps:
        swaps = []
        adds = []
        rems = []
        counts = []

        count = 0
        while heap[0] == (1,1):
            heappop(heap)
            count += 1
        counts.append((1,count))

        while len(heap) > 0:
            change = heappop(heap)
            if len(heap) > 0 and change[0] == heap[0][0]:
                heappop(heap)
                swaps.append(change[0])
            elif change[1] == 1:
                adds.append(change[0])
                count += 1
                counts.append((change[0], count))
            elif change[1] == 0:
                rems.append(change[0])
                count -= 1
                counts.append((change[0], count))


        results.append({"swaps": swaps,
                        "adds": adds,
                        "removes": rems,
                        "counts": counts,
                        }
                       )
    
    lambdas = [[(p[0],p[1]) for p in l_seq ] for l_seq in seq_dat["lambda"]]
    
    all_results = {"parents": results,
                   "lambdas": lambdas
                  }
    return all_results


def plot_summary(summary, plot_file):

    plt.figure(figsize=(12.0, 4.0))

    lambdas = summary["lambdas"]
    summary = summary["parents"]   
 
    #for d in summary:
    #    t = [p[0] for p in d["counts"]]
    #    n = [p[1] for p in d["counts"]]
    #    plt.plot(t, n, linewidth=0.25)

    #for l_seq in lambdas:
        #l_t = [p[0] for p in l_seq]
        #l_l = [[p[1] for p in l_seq] for l_seq in lambdas]
        #plt.plot(l_t, l_l, linewidth=2.0)

    l_l = [[p[1] for p in l_seq] for l_seq in lambdas]
    plt.violinplot(l_l, showmeans=True, showextrema=True)

    #plt.xlim(25000, 30000.0)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    return


if __name__=="__main__":

    print(sys.argv[1])
    sequence_files = [p for arg in sys.argv[1:] for p in glob.glob(arg)]
    print(sequence_files)
    #output_file = sys.argv[2]

    for sf in sequence_files:
        with open(sf, "r") as f:
            seq_dat = json.load(f)

        output_dict = summarize_sequences(seq_dat)

        kvs = su.parse_path_kvs(sf)
        kvs["chain"] = os.path.basename(sf).split(".")[0]
        print("\t",kvs)
        plot_file = "_".join(["{}={}".format(k,v) for k,v in kvs.items()]) + ".png"
        print("Plotting: ", plot_file)
        plot_summary(output_dict, plot_file)

    #with open(output_file, "w") as f:
    #    json.dump(output_dict, f)

    
