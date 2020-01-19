# postprocess_samples.jl
# 2020-01-17
# David Merrell
# 
# Functions for postprocessing MCMC sample output.
# Specifically for computing convergence statistics
# (PSRF and effective samples).

module SamplePostprocess

using JSON


function load_samples(samples_path::String)

    f = open(samples_path, "r")
    str = read(f, String)
    close(f)
    d = JSON.parse(str)

    return d["parent_sets"], d["lambda"], d["n"]
    
end


"""
Given a vector of changepoints and a sorted vector of endpoints,
return the sums of the sequences terminating at the endpoints. 
"""
function changepoint_sum(changepoints::Vector, endpoints::Vector{Int64})
  
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



function make_lambda_means(lambda_samples::Vector; stepsize::Int64=100)

end



end # END MODULE
