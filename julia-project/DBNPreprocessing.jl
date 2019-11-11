
protein_names = CSV.read("data/protein_names.csv");
protein_vec = convert(Matrix, protein_names)[:,1];
protein_vec = [name[3:length(name)-2] for name in protein_vec]
reference_adjacency = CSV.read("data/prior_graph.csv");
adj_mat = convert(Matrix, reference_adjacency);
timesteps = CSV.read("data/time.csv");
timeseries_data = CSV.read("data/mukherjee_data.csv");

function timeseries_preprocess!(timeseries_df)
    timeseries_data.condition = [ split(s, " ")[2] for s in timeseries_data[!, :Column1] ]
    timeseries_data.time = [ parse(Float64, split(s, " ")[4][2:end]) for s in timeseries_data[!, :Column1]]
    timeseries_data.replicate = [tryparse(Int64, split(s, " ")[6][2:end-1]) for s in timeseries_data[!, :Column1]]
    
    gp = DataFrames.groupby( timeseries_data, [:condition; :replicate])
    ts_vec = [convert(Matrix, g)[:,2:end-3] for g in gp]
    ts_vec = [convert(Matrix{Float64}, m) for m in ts_vec]
    
    return ts_vec
end

function build_reference_graph(vertices::Vector{T}, reference_adj::Array{Int,2}) where T
    dg = DiGraph{T}()
    for i=1:size(reference_adj)[1]
        for j=1:size(reference_adj)[2]
            if reference_adj[i,j] == 1
                add_edge!(dg, vertices[i], vertices[j])
            end
        end
    end
    return dg
end;
