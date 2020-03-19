# scoring.jl
# 2019-12-28
# David Merrell
#
# Methods for scoring performance
# in network inference tasks.

module Scoring

using JSON, CSV, ArgParse 

export auroc, auprc, julia_main

"""
Compute AUROC from the given predictions,
with respect to the given ground-truth.
Integral is computed via trapezoidal rule.

Has n*log(n) complexity 
(superlinear expense incurred by sorting; 
 the AUROC loop has linear expense).
"""
function auroc(preds::Vector, truths::Vector{Bool}; return_curve::Bool=false)

    if return_curve
        curve = Vector{Vector{Float64}}()
        push!(curve, [0.0;0.0])
    end

    # Sort the predictions (and corresponding ground truth)
    # by confidence level
    srt_inds = sortperm(preds, order=Base.Order.Reverse)
    srt_preds = preds[srt_inds]
    srt_truths = truths[srt_inds]
    
    N = length(srt_preds)
    npos = sum(truths)
    nneg = N - npos

    area = 0.0
    tp = 0
    fp = 0
    prev_fp = 0
    prev_tp = 0

    # Loop through the predictions 
    # (in order of decreasing confidence level)
    for i=1:N
        if (i > 1) && (srt_preds[i] != srt_preds[i-1]) 
            area += 0.5*(prev_tp + tp)*(fp - prev_fp)/(npos*nneg)
            if return_curve
                push!(curve, [fp/nneg; tp/npos])
            end
            prev_tp = tp
            prev_fp = fp
        end
        
        if srt_truths[i]
            tp += 1
        else
            fp += 1
        end
    end 
    area += 0.5*(prev_tp + tp)*(fp - prev_fp)/(npos*nneg)
    
    if return_curve
        push!(curve, [1.0; 1.0])
        return area, transpose(hcat(curve...))
    else
        return area
    end
    
end


"""
Compute AUPRC from the given predictions,
with respect to the given ground-truth.
Integral is computed via trapezoidal rule.

Has n*log(n) complexity 
(superlinear expense incurred by sorting; 
 the AUPRC loop has linear expense).
"""
function auprc(preds::Vector, truths::Vector{Bool}; return_curve::Bool=false)

    if return_curve
        curve = Vector{Vector{Float64}}()
        push!(curve, [0.0;1.0])
    end
    
    # Sort the predictions (and corresponding ground truth)
    # by confidence level
    srt_inds = sortperm(preds, order=Base.Order.Reverse)
    srt_preds = preds[srt_inds]
    srt_truths = truths[srt_inds]

    area = 0.0
    tp = 0
    fp = 0
    prev_p = 1.0
    prev_r = 0.0
    N = length(srt_truths)
    npos = sum(srt_truths)
    nneg = N - npos
    
    for i=1:N
        if (i>1) && (srt_preds[i] < srt_preds[i-1])
            
            recall = tp / npos
            precision = tp / (tp + fp)
            if return_curve
                push!(curve, [recall; precision])
            end
            area += 0.5*(precision + prev_p)*(recall - prev_r)

            prev_r = recall
            prev_p = precision
        end

        if srt_truths[i]
            tp += 1
        else
            fp += 1
        end
    end

    recall = tp / npos
    precision = tp / (tp + fp)
    area += 0.5*(precision + prev_p)*(recall - prev_r)

    if return_curve
        push!(curve, [recall; precision])
        return area, transpose(hcat(curve...))
    end

    return area
end


"""
Take a "vector of vectors" representation of the
network edges and return a single vector in canonical order.
"""
function ps_to_vec(parent_sets::Vector)
    return [e for ps in parent_sets for e in ps]
end


"""
Read a JSON file containing edge predictions.
We assume there's a "edge_conf_key" field specifying
where to access the predicted parent sets.

Return a single vector of confidence levels for all edges.
"""
function load_ps_predictions(pred_fname)

    f = open(pred_fname, "r")
    js_d = JSON.parse(read(f, String))
    close(f)

    key = js_d["edge_conf_key"]

    parent_sets = js_d[key]

    return ps_to_vec(parent_sets)
end


"""
Read a CSV file containing a true adjacency matrix.
We assume the file is comma-delimited with no header
or index column; the (i,j)-th entry denotes the jth
parent of the ith vertex (that is: the columns are
parent sets). Each entry must be 0 or 1.
"""
function load_ps_truths(truths_fname)

    df = CSV.read(truths_fname; header=false)
    adj_mat = convert(Matrix{Bool}, df)

    parent_sets = [adj_mat[:,i] for i=1:size(adj_mat,2)]

    return ps_to_vec(parent_sets)
end

"""
Given 
* a vector of edge prediction filenames
* a vector of edge truth filenames
* a scoring function;
return a vector of the scores 
"""
function edge_score(pred_fname::String, 
                    truth_fname::String,
                    score_fn::Function)

    preds = load_ps_predictions(pred_fname)
    truths = load_ps_truths(truth_fname)
    return score_fn(preds, truths)

end


function get_args(args::Vector{String})

    s = ArgParseSettings()
    @add_arg_table s begin
        "--pred-file"
            help="The path to a JSON file containing edge predictions"
            required=true
            arg_type=String
        "--truth-file"
            help="The path to a CSV file containing true adjacency matrice(s)."
            required=true
            arg_type=String
        "--output-file"
            help="The path of an output JSON file"
            required=true
            arg_type=String
        "--skip-roc"
            help="Do NOT compute ROC or AUROC for the predictions."
            action=:store_true
    end

    args = parse_args(args, s)
    return args
end


# Main function -- for purposes of static compilation
function julia_main()

    args = get_args(ARGS)

    pred_fname = args["pred-file"]
    truth_fname = args["truth-file"]
    output_fname = args["output-file"]
    skip_roc = args["skip-roc"]

    result = Dict{String,Any}()
    
    # Compute AUCPR and PR curve 
    pr_score_fn = (pred, truth) -> auprc(pred, truth; return_curve=true)
    pr_stuff = edge_score(pred_fname, truth_fname, pr_score_fn) 
    result["aucpr"] = pr_stuff[1]
    result["pr_curve"] = pr_stuff[2] 

    if !skip_roc
        # Compute AUCROC and ROC curve for each set of predictions
        roc_score_fn = (pred, truth) -> auroc(pred, truth; return_curve=true)
        roc_stuff = edge_score(pred_fname, truth_fname, roc_score_fn) 
        result["aucroc"] = roc_stuff[1] 
        result["roc_curve"] = roc_stuff[2] 
    end

    f = open(output_fname, "w")
    JSON.print(f, result)
    close(f)

    return 0
end


# END MODULE
end

using .Scoring

julia_main()


