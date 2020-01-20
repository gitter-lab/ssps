# postprocess_samples.jl
# 2020-01-17
# David Merrell
# 
# Functions for postprocessing MCMC sample output.
# Specifically for computing convergence statistics
# (PSRF and effective samples).

module SamplePostprocess

using JSON


"""
Read an MCMC samples output file
"""
function load_samples(samples_file_path::String)

    f = open(samples_file_path, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    return d["parent_sets"], d["lambda"], d["n"]
end


"""
Select the subsequence starting at `start_t` and ending at `end_t` -- *inclusive*
"""
function changepoint_select(changepoints::Vector, start_t::Int64, end_t::Int64)

    start_idx = searchsortedfirst(changepoints, start_t; by=x->x[1])
    end_idx = searchsortedlast(changepoints, end_t; by=x->x[1])

    result = changepoints[start_idx:end_idx]

    if changepoints[start_idx][1] != start_t
        prev_v = 0
        if start_idx > 1
            prev_v = changepoints[start_idx - 1][2]
        end
        pushfirst!(result, (start_t, prev_v))
    end

    return result
end



"""
Apply a function `op` to each entry of the sequence defined by
the changepoint vector.
"""
function changepoint_map(op::Function, changepoints::Vector)
    result = [(i, op(cp)) for cp in changepoints]
end


"""
Given a vector of changepoints and a vector of endpoints,
return the means of the sequences terminating at the endpoints.
"""
function sequence_means(changepoints::Vector, 
                        endpoints::Vector{Int64})

    sequence_sums = changepoint_sum(changepoints, endpoints)
    
    start_t = changepoints[1][1]
 
    return [s/(endpoints[i] - start_t + 1) for (i, s) in enumerate(sequence_sums)]
end


"""
Given a vector of changepoints and a vector of endpoints,
return the variances of the sequences terminating at the endpoints.
"""
function sequence_variances(changepoints::Vector,
                            endpoints::Vector{Int64})
    
    means = sequence_means(changepoints, endpoints)
    
    for mean in means
        
    end
end


"""
Given a vector of changepoints and a sorted vector of endpoints,
return the sums of the sequences terminating at the endpoints. 
"""
function changepoint_sum(changepoints::Vector, 
                         endpoints::Vector{Int64})
  
    cur_t = min(changepoints[1][1], endpoints[1])
    cur_v = 0.0
    cur_changepoint = 1
    cur_endpoint = 1
    cur_sum = 0
    sums = zeros(length(endpoints))

    while cur_endpoint <= length(endpoints)
        
        # is our current point a changepoint?
        if cur_changepoint <= length(changepoints) && cur_t == changepoints[cur_changepoint][1]
            cur_v = changepoints[cur_changepoint][2]
            cur_changepoint += 1
        end
        
        # is our current point an endpoint?
        if cur_t == endpoints[cur_endpoint]
            sums[cur_endpoint] = cur_sum + cur_v
            cur_endpoint += 1
        end
        if cur_endpoint > length(endpoints)
            break
        end

        # Get the next point of interest
        if cur_changepoint > length(changepoints)
            next_t = endpoints[cur_endpoint]
        else
            next_t = min(changepoints[cur_changepoint][1], endpoints[cur_endpoint])
        end
        
        #increment sum, update state
        cur_sum += cur_v * (next_t - cur_t)
        cur_t = next_t
    end
    
    return sums
end



end # END MODULE
