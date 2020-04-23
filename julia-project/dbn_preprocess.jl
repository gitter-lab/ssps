# dbn_preprocess.jl
# 2019-11-12
# David Merrell
#
#

using CSV
using DataFrames
using Statistics


function ingest_timeseries!(timeseries_df)

    result = Vector{Matrix{Float64}}()

    #N = convert(Int64, maximum(timeseries_df.timeseries))

    for name in unique(timeseries_df.timeseries)
        ts = timeseries_df[timeseries_df.timeseries .== name, :]
        sort!(ts, [:timestep])
        push!(result, convert(Matrix{Float64}, ts[:,3:end]))
    end

    return result
end


function ingest_conf_adjacency!(adjacency_df)

    parent_confs = Vector{Dict{Int64,Float64}}()
    adj_mat = convert(Matrix{Float64}, adjacency_df)

    for j=1:size(adj_mat,2)
        ps = Dict{Int64,Float64}()
        for i=1:size(adj_mat,1)
            if adj_mat[i,j] > 0.0
                ps[i] = adj_mat[i,j]
            end 
        end
        push!(parent_confs, ps)
    end

    return parent_confs
end


"""
    load_formatted_data(timeseries_data_path::String,
                        reference_graph_path::String;
                        boolean_adj::Bool=true)

Load data from files and convert to the types requisite
for our probabilistic program.

This function assumes the data (time series and adjacency matrices)
are stored in text-delimited files formatted in a very particular way.
"""
function load_formatted_data(timeseries_data_path::String,
		                     reference_graph_path::String)

    timeseries_df = CSV.read(timeseries_data_path)
    ts_vec = ingest_timeseries!(timeseries_df)

    ref_adj_df = CSV.read(reference_graph_path, header=false)
    ref_ps = ingest_conf_adjacency!(ref_adj_df)

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

    return Xminus, Xplus
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

    return Xminus, Xplus_v

end


