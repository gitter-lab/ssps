# dbn_preprocessing.jl
# 2019-11-12
# David Merrell
#
#

using CSV
using DataFrames

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
    ref_dg = build_reference_graph(protein_vec, adj_mat)

    timesteps = CSV.read(timesteps_path)

    return (timeseries_vec, protein_vec, ref_dg, timesteps)
end
