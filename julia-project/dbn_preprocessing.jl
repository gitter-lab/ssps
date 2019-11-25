# dbn_preprocessing.jl
# 2019-11-12
# David Merrell
#
#

using CSV
using DataFrames
using Statistics

function timeseries_preprocess!(timeseries_df)
    timeseries_df.condition = [ split(s, " ")[2] for s in timeseries_df[!, :Column1] ]
    timeseries_df.time = [ parse(Float64, split(s, " ")[4][2:end]) for s in timeseries_df[!, :Column1]]
    timeseries_df.replicate = [tryparse(Int64, split(s, " ")[6][2:end-1]) for s in timeseries_df[!, :Column1]]

    gp = DataFrames.groupby( timeseries_df, [:condition; :replicate])
    ts_vec = [convert(Matrix, g)[:,2:end-3] for g in gp]
    ts_vec = [convert(Matrix{Float64}, m) for m in ts_vec]

    return ts_vec
end

function build_reference_graph(vertices::Vector{T}, reference_adj::Array{Int,2}) where T
    dg = PSDiGraph{T}()
    for i=1:size(reference_adj)[1]
        for j=1:size(reference_adj)[2]
            if reference_adj[i,j] == 1
                add_edge!(dg, vertices[i], vertices[j])
            end
        end
    end
    return dg
end


function vectorize_reference_adj(reference_adj)
    return [convert(Vector{Bool}, reference_adj[:,i]) for i=1:size(reference_adj)[1]]
end


function hill_2012_preprocess(timeseries_data_path,
                              protein_names_path,
                              prior_graph_path,
                              timesteps_path)

    timeseries_data = CSV.read(timeseries_data_path)
    timeseries_vec = timeseries_preprocess!(timeseries_data)

    protein_names = CSV.read(protein_names_path)
    protein_vec = convert(Matrix, protein_names)[:,1]
    protein_vec = [name[3:length(name)-2] for name in protein_vec]

    reference_adjacency = CSV.read(prior_graph_path)
    adj_mat = convert(Matrix, reference_adjacency)
    #ref_dg = build_reference_graph(protein_vec, adj_mat)
    reference_adj = vectorize_reference_adj(adj_mat)

    timesteps = CSV.read(timesteps_path)

    return (timeseries_vec, protein_vec, reference_adj, timesteps)
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
    Xminus_stacked = fill(Xminus,size(Xminus)[2])

    return Xminus_stacked, Xplus_v

end


