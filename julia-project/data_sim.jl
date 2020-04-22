
module DBNDataSim

using Statistics
using Distributions
using Random
using Combinatorics
import LinearAlgebra: dot
using CSV
using DataFrames

export generate_random_digraph, modify_and_simulate, save_dataset, save_graph 

"""
    generate_random_digraph(V::Int64, p::Float64)

Generate a random directed graph with `V` vertices.
Draws from a uniform distribution over directed graphs;
every edge has probability `p` of being in the graph. 
"""
function generate_random_digraph(V::Int64, p::Float64)
    parent_sets = [convert(Vector{Bool}, rand(Distributions.Bernoulli(p), V)) for i=1:V]
    return parent_sets
end


"""
    digraph_prior(lambda::Float64, reference::Vector{Vector{Bool}})

Sample a new graph G from the density

P(G) \\propto exp(-lambda * |G \\ G'|)

Where G' is a reference graph.
"""
function digraph_prior(lambda::Float64, reference::Vector{Vector{Bool}})

    result = []
    prob = exp(-lambda) / (1.0 + exp(-lambda))
    for (i, ps) in enumerate(reference)
        new_ps = Vector{Bool}()
        for (j, parent) in enumerate(ps)
            if parent
                bp = 0.5
            else
                bp = prob
            end
            push!(new_ps, rand(Distributions.Bernoulli(bp)))
        end
        push!(result, new_ps)
    end
    return result
end


"""
    modify_digraph!(ps::Vector{Vector{Int64}}, keep::Int64, add::Int64)

Modify a given graph
"""
function modify_digraph!(ps::Vector{Vector{Bool}}, remove::Float64, add::Float64)

    # get the indices of the original edges
    ps_idx = []
    not_ps_idx = [] # and the complement
    for i=1:length(ps)
        for (j, b) in enumerate(ps[i])
            if b
                push!(ps_idx, (i,j))
            else
                push!(not_ps_idx, (i,j))
            end
        end
    end
        
    # Randomly remove edges from the graph
    n_edges = length(ps_idx)
    keep = Int64(round((1.0 - remove)*n_edges))
    to_remove = Random.randperm(n_edges)[1:max(n_edges - keep, 0)]
    for rm_idx in to_remove
        (i,j) = ps_idx[rm_idx]
        ps[i][j] = false
    end
    
    # Randomly add edges (from the complement)
    n_not_edges = length(not_ps_idx)
    add = Int64(round(add*n_edges))
    to_add = Random.randperm(n_not_edges)[1:min(add, n_not_edges)]
    for add_idx in to_add
        (i,j) = not_ps_idx[add_idx]
        ps[i][j] = true
    end
    
    return ps
end


"""
    n_coeffs(V::Int64, max_degree::Int64)

Given a number of variables (V) and a maximum polynomial degree
for the regression model, compute the number of coefficients required
by the model. (this includes the constant term)
"""
function n_coeffs(V::Int64, max_degree::Int64)
    return sum([binomial(V, i) for i=0:min(V, max_degree)])
end
    


"""
    generate_reg_coeffs(ps::Vector{Vector{Bool}}, coeff_std::Float64, regression_deg::Int64)

Given a vector of parent sets, generate a vector of regression coefficient vectors.
(This includes the constant term)
"""
function generate_reg_coeffs(ps::Vector{Vector{Bool}}, coeff_std::Float64, regression_deg::Int64)
   
    result = Vector{Vector{Float64}}()
    for v in ps
        num_vars = sum(v)
        bw = n_coeffs(num_vars, regression_deg)
        push!(result, coeff_std.*randn(bw))
    end
    
    return result
end


"""
    initialize_time_series(V)
"""
function initialize_time_series(V::Int64; init_std::Float64=1.0)
    return init_std.*randn(V)
end


"""
    compute_B_data(x_vec::Vector{Float64}, ps::Vector{Bool}, regression_deg::Int64)
"""
function compute_B_data(x_vec::Vector{Float64}, ps::Vector{Bool}, regression_deg::Int64)
   
    used_x = x_vec[ps]
    num_parents = length(used_x)
    
    B_vec = ones(n_coeffs(num_parents, regression_deg))
    
    col = 2
    for deg=1:regression_deg
        for comb in Combinatorics.combinations(used_x, deg)
            B_vec[col] = prod(comb)
            col += 1
        end
    end
    
    return B_vec
end

"""
    generate_next_timestep_data(x_vec::Vector{Float64}, parent_sets::Vector{Vector{Bool}}, regression_std::Int64)
"""
function generate_next_timestep_data(x_vec::Vector{Float64}, 
                                     parent_sets::Vector{Vector{Bool}},
                                     regression_coeffs::Vector{Vector{Float64}},
                                     regression_deg::Int64,
                                     regression_std::Float64)
    
    next_x = Vector{Float64}()
    
    for (i, ps) in enumerate(parent_sets)
        
        data_vec = compute_B_data(x_vec, ps, regression_deg)
        
        push!(next_x, dot(data_vec, regression_coeffs[i]) + regression_std*randn() )
    end
    
    return next_x
end


"""
    generate_time_series(parent_sets::Vector{Vector{Bool}}, 
                         V::Int64, T::Int64)

Create a simulated time series with `T` time steps
"""
function generate_time_series(parent_sets::Vector{Vector{Bool}},
                              T::Int64,
                              regression_coeffs::Vector{Vector{Float64}},
                              regression_deg::Int64,
                              regression_std::Float64)
    
    result = zeros(T, length(parent_sets))
    result[1,:] = initialize_time_series(length(parent_sets); init_std=regression_std)
    #result[1,:] = initialize_time_series(length(parent_sets))
    
    for t=2:T
        result[t,:] = generate_next_timestep_data(result[t-1,:], parent_sets,
                                                  regression_coeffs,
                                                  regression_deg,
                                                  regression_std)
    end
    
    return result
end


"""
    generate_dataset(T::Int64, N::Int64, parent_sets::Vector{Vector{Bool}},
                     coeff_std::Float64, regression_deg::Int64, regression_std::Float64)

Given a graph structure, generate some coefficients and simulate a dataset of N time series,
each time series over T timesteps.
"""
function generate_dataset(T::Int64, N::Int64,
                          parent_sets::Vector{Vector{Bool}},
                          coeff_std::Float64, 
                          regression_deg::Int64, regression_std::Float64)
    
    coeffs = generate_reg_coeffs(parent_sets, coeff_std, regression_deg)
    
    return [generate_time_series(parent_sets, T, coeffs, regression_deg, regression_std) for i=1:N]
    
end



"""
    modify_and_simulate(ref_ps::Vector{Vector{Bool}}, remove::Int64, add::Int64)

Given a reference graph, create a modified version of it 
and then simulate a dataset from that modified graph.

Modify the graph via `remove`ing a fraction of the current edges 
and `add`ing a fraction of new edges.
"""
function modify_and_simulate(ref_ps::Vector{Vector{Bool}}, remove::Float64, add::Float64, 
                             T::Int64, N::Int64,
                             coeff_std::Float64, 
                             regression_deg::Int64, regression_std::Float64)
   
    V = size(ref_ps, 1)
    ref_ps_copy = [copy(ps) for ps in ref_ps]
    modify_digraph!(ref_ps_copy, remove, add)
    
    ds = generate_dataset(T, N, ref_ps, coeff_std, regression_deg, regression_std) 
    
    return ref_ps_copy, ds
end

"""
    save_dataset(dataset::Vector{Array{Float64,2}}, file_name::String)
"""
function save_dataset(dataset::Vector{Array{Float64,2}}, file_name::String)

    V = size(dataset[1],2)
    ndigs = length(string(V))

    df = DataFrames.DataFrame(Dict([:timeseries=>Int[]; :timestep=>Int[]; 
				    [(Symbol("var",lpad(i,ndigs,"0"))=>Float64[]) for i=1:V]]))    
    
    for (i, timeseries) in enumerate(dataset)
        for t=1:size(timeseries,1)
            push!(df, [i; t; dataset[i][t,:]])
        end
    end

    CSV.write(file_name, df; delim="\t")
end
    

"""
    save_graph(parent_sets::Vector{Vector{Float64}}, file_name::String)
"""
function save_graph(parent_sets::Vector{Vector{Bool}}, file_name::String)
   
    CSV.write(file_name, 
              DataFrame(convert(Matrix{Int64}, hcat(parent_sets...)));
              delim=",", writeheader=false)
    
end

end
