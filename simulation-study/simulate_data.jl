
module Simulator

include("DBNDataSim.jl")
using .DBNDataSim

# Defining this function allows us to statically compile the
# simulator via PackageCompiler.jl (big speedup)
Base.@ccallable function julia_main(ARGS::Vector{String})::Cint

    V = parse(Int64, ARGS[1])
    T = parse(Int64, ARGS[2])
    p = parse(Float64, ARGS[3])
    N = 1
    remove = parse(Float64, ARGS[4])
    add = parse(Float64, ARGS[5])
    regression_deg = parse(Int64, ARGS[6])
    ref_dg_filename = ARGS[7]
    true_dg_filename = ARGS[8]
    timeseries_filename = ARGS[9]
    coeff_std = 1.0/sqrt(V)
    regression_std = 1.0/sqrt(T)
    
    ref_dg = generate_random_digraph(V,p)
    
    true_dg, timeseries = modify_and_simulate(ref_dg, remove, add,
    					  T, N, coeff_std,
    					  regression_deg,
    					  regression_std)
    
    save_graph(ref_dg, ref_dg_filename)
    save_graph(true_dg, true_dg_filename)
    save_dataset(timeseries, timeseries_filename)

    return 0

end

end

