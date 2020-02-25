
module Simulator

include("data_sim.jl")
using .DBNDataSim

export julia_main

function julia_main()

    ref_dg = nothing
    true_dg = nothing
    timeseries = nothing
        
    V = parse(Int64, ARGS[1])
    T = parse(Int64, ARGS[2])
    p = 5.0/V # 5 parents on average
    N = parse(Int64, ARGS[3]) 
    remove = parse(Float64, ARGS[4])
    add = parse(Float64, ARGS[5])
    regression_deg = parse(Int64, ARGS[6])
    ref_dg_filename = ARGS[7]
    true_dg_filename = ARGS[8]
    timeseries_filename = ARGS[9]
    coeff_std = 1.0/sqrt(V)
    regression_std = 1.0/sqrt(T)/1000.0 # TODO: THIS NEEDS TO BE SMALLER.
                                 #       (more signal, less noise)

    while true
        
        ref_dg = generate_random_digraph(V,p)
        
        true_dg, timeseries = modify_and_simulate(ref_dg, remove, add,
        					  T, N, coeff_std,
        					  regression_deg,
        					  regression_std)
        
        if maximum([maximum(ts) for ts in timeseries]) < 100
            break
        end
    end    
        
    save_graph(ref_dg, ref_dg_filename)
    save_graph(true_dg, true_dg_filename)
    save_dataset(timeseries, timeseries_filename)

    return 0

end

end

using .Simulator

julia_main()


