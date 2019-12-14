# perform_mcmc.jl
# 2019-12-12
# David Merrell
#
# This script invokes Catsupp to infer a network
# from time series data. The command line interface
# is rudimentary -- it's mostly intended for our
# Snakemake-based analysis framework.


include("Catsupp.jl")
import .Catsupp: parse_script_arguments, 
                 perform_inference, make_output_json 
 
arg_vec = parse_script_arguments()
perform_inference(arg_vec...)

