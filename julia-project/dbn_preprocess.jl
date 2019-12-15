# dbn_preprocess.jl
# 2019-11-12
# David Merrell
#
#

using CSV
using DataFrames
using Statistics


function hill_timeseries_preprocess!(timeseries_df)
    
    # Extract identifier information from the text column
    timeseries_df.condition = [ split(s, " ")[2] for s in timeseries_df[!, :Column1] ]
    timeseries_df.time = [ parse(Float64, split(s, " ")[4][2:end]) for s in timeseries_df[!, :Column1]]
    timeseries_df.replicate = [tryparse(Int64, split(s, " ")[6][2:end-1]) for s in timeseries_df[!, :Column1]]

    # group by condition and timestep; take the mean over the replicates.
    ts_vec = Vector{Matrix{Float64}}()
    condition_gps = DataFrames.groupby(timeseries_df, [:condition])
    for c_g in condition_gps
        ts_gps = DataFrames.groupby(c_g, [:time])
        ts = zeros(length(ts_gps), size(timeseries_df,2)-4)
        for (i, t_g) in enumerate(ts_gps)
            ts[i,:] = mean(convert(Matrix{Float64}, t_g[:,2:end-3]), dims=1)
        end
        push!(ts_vec, ts)
    end
    
    # It turns out this timeseries data needs to be log-transformed for normality
    ts_vec = [log.(m) for m in ts_vec]

    return ts_vec
end


function standard_timeseries_preprocess!(timeseries_df)

    result = Vector{Matrix{Float64}}()

    N = convert(Int64, maximum(timeseries_df.timeseries))

    for i=1:N
        ts = timeseries_df[timeseries_df.timeseries .== i, :]
        sort!(ts, [:timestep])
        push!(result, convert(Matrix{Float64}, ts[:,3:end]))
    end

    return result
end


"""

Receive a dataframe containing an adjacency matrix,
and return a Vector{Vector{Bool}} representing the 
parent sets of the graph.
"""
function standard_adjacency_preprocess!(adjacency_df)

    parent_sets = Vector{Vector{Bool}}()

    adj_mat = convert(Matrix{Bool}, adjacency_df)
    for j=1:size(adj_mat,2)
        push!(parent_sets, adj_mat[:,j])
    end

    return parent_sets
end



#function build_reference_graph(vertices::Vector{T}, reference_adj::Array{Int,2}) where T
#    dg = PSDiGraph{T}()
#    for i=1:size(reference_adj)[1]
#        for j=1:size(reference_adj)[2]
#            if reference_adj[i,j] == 1
#                add_edge!(dg, vertices[i], vertices[j])
#            end
#        end
#    end
#    return dg
#end

"""
hill_2012_preprocess(timeseries_data_path,
                     protein_names_path,
                     prior_graph_path,
                     timesteps_path)

Load and preprocess the data from the Hill et al. 2012
paper.
"""
function hill_2012_preprocess(timeseries_data_path,
                              protein_names_path,
                              prior_graph_path,
                              timesteps_path)

    timeseries_data = CSV.read(timeseries_data_path)
    timeseries_vec = hill_timeseries_preprocess!(timeseries_data)

    protein_names = CSV.read(protein_names_path)
    protein_vec = convert(Matrix, protein_names)[:,1]
    protein_vec = [name[3:length(name)-2] for name in protein_vec]

    reference_adjacency = CSV.read(prior_graph_path)
    reference_adj = standard_adjacency_preprocess!(reference_adjacency)

    timesteps = CSV.read(timesteps_path)

    return (timeseries_vec, protein_vec, reference_adj, timesteps)
end


"""
    load_simulated_data(timeseries_data_path::String,
                        reference_graph_path::String)

Load data from files and convert to the types requisite
for our probabilistic program.

This function assumes the data (time series and adjacency matrices)
are stored in text-delimited files of the standard format.
"""
function load_simulated_data(timeseries_data_path::String,
		             reference_graph_path::String)

    timeseries_df = CSV.read(timeseries_data_path)
    ts_vec = standard_timeseries_preprocess!(timeseries_df)

    ref_adj_df = CSV.read(reference_graph_path, header=false)
    ref_ps = standard_adjacency_preprocess!(ref_adj_df)
   
    return ts_vec, ref_ps
end


function combine_X(X::Vector{Array{Float64,2}})

    len = sum([size(x)[1] - 1 for x in X])
    wid = size(X[1])[2]

    Xminus = zeros(len, wid)
    Xplus = zeros(len, wid)

    l_ind = 1
    for x in X
        r_ind = l_ind + size(x)[1] - 2
        Xminus[l_ind:r_ind, :] = x[1:size(x)[1]-1, :]
        Xplus[l_ind:r_ind, :] = x[2:size(x)[1], :]
        l_ind = r_ind + 1
    end

    return Xminus, standardize_X(Xplus)[1]
end

function standardize_X(X_arr)

    mus = Statistics.mean(X_arr, dims=1)
    sigmas = Statistics.std(X_arr, dims=1)

    X_arr =  (X_arr .- mus) ./ sigmas

    return X_arr, mus, sigmas
end

function vectorize_X(Xminus::Array{Float64,2}, 
		     Xplus::Array{Float64,2})
    
    Xplus_v = [Xplus[:,i] for i=1:size(Xplus)[2]]
    Xminus_stacked = fill(Xminus,size(Xminus)[2])

    return Xminus_stacked, Xplus_v

end


