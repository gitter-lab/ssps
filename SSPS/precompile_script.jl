using SSPS

push!(ARGS, "../../run_ssps/example_timeseries.csv")
push!(ARGS, "../../run_ssps/example_prior.csv")
push!(ARGS, "dumb_output.json")
push!(ARGS, "60")

julia_main()

