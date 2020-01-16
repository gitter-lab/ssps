# Snakefile
# 2019-12
# David Merrell
#
# This snakemake file manages the execution of all 
# analyses related to the Merrell et al. 
# ISMB 2020 submission.
#
# Run as follows:
# $ cd directory/containing/Snakefile
# $ snakemake
#
# See repository README.md for more details.

import glob
import os

###########################
# IMPORT CONFIG FILE
###########################
configfile: "analysis_config.yaml"

###########################
# DEFINE SOME VARIABLES 
###########################

# directories
ROOT_DIR = os.getcwd()
BIN_DIR = os.path.join(ROOT_DIR,"bin")
FIG_DIR = os.path.join(ROOT_DIR,"figures")
JULIA_PROJ_DIR = os.path.join(ROOT_DIR, "julia-project")
HILL_DIR = os.path.join(ROOT_DIR, "hill-method")
FUNCH_DIR = os.path.join(ROOT_DIR, "funchisq")
TEMP_DIR = config["temp_dir"]
EXP_DATA_DIR = os.path.join(ROOT_DIR,"experimental_data")

# simulation study directories
SIM_DIR = os.path.join(TEMP_DIR, "simulation_study")
SIMDAT_DIR = os.path.join(SIM_DIR, "datasets")
TS_DIR = os.path.join(SIMDAT_DIR, "timeseries")
REF_DIR = os.path.join(SIMDAT_DIR, "ref_graphs")
TRU_DIR = os.path.join(SIMDAT_DIR, "true_graph")
RAW_DIR = os.path.join(SIM_DIR, "raw")
PRED_DIR = os.path.join(SIM_DIR, "predictions")
SCORE_DIR = os.path.join(SIM_DIR, "scores")

# Simulation study parameters
SIM_PARAMS = config["simulation_study"]
SIM_TIMEOUT = SIM_PARAMS["timeout"]
SIM_REPLICATES = list(range(SIM_PARAMS["N"]))
SIM_GRID = SIM_PARAMS["simulation_grid"]
SIM_M = SIM_GRID["M"]
POLY_DEG = SIM_PARAMS["polynomial_degree"]

# MCMC hyperparameters (for simulation study)
MC_PARAMS = SIM_PARAMS["mcmc_hyperparams"]
REG_DEGS = MC_PARAMS["regression_deg"]
BURNIN = MC_PARAMS["burnin"]

# Hill hyperparameters
HILL_PARAMS = SIM_PARAMS["hill_hyperparams"]
HILL_MAX_INDEG = HILL_PARAMS["max_indeg"]
HILL_TIME_PARAMS = config["hill_timetest"]
HILL_TIME_COMBS = HILL_TIME_PARAMS["deg_v_combs"]
HILL_MODES = HILL_TIME_PARAMS["modes"]
HILL_TIME_TIMEOUT = HILL_TIME_PARAMS["timeout"]

# Convergence analysis 
CONV_DIR = os.path.join(TEMP_DIR, "convergence")
CONV_RES_DIR = os.path.join(CONV_DIR, "results")
CONV_RAW_DIR = os.path.join(CONV_DIR, "raw")
CONV_PARAMS = config["convergence_analysis"]
CONV_SIM_GRID = CONV_PARAMS["simulation_grid"]
CONV_DATASETS = CONV_PARAMS["experimental_datasets"]
CONV_DEGS = CONV_PARAMS["mcmc_hyperparams"]["regression_deg"]
CONV_REPLICATES = list(range(CONV_PARAMS["N"]))
CONV_CHAINS = list(range(CONV_PARAMS["n_chains"]))
CONV_MAX_SAMPLES = CONV_PARAMS["max_samples"]
CONV_TIMEOUT = CONV_PARAMS["timeout"]


#############################
# RULES
#############################
rule all:
    input:
        # convergence tests on simulated data
        expand(CONV_RAW_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}/mcmc_d={d}/chain_{c}.bson",
               v=CONV_SIM_GRID["V"], r=CONV_SIM_GRID["R"], a=CONV_SIM_GRID["A"],
               t=CONV_SIM_GRID["T"], rep=CONV_REPLICATES, d=CONV_DEGS, c=CONV_CHAINS) 
        # convergence test results on experimental data
        #expand(CONV_RAW_DIR+"/{dataset}/mcmc_d={d}/chain_{c}.bson", 
        #       ds=CONV_DATASETS, d=CONV_DEGS, c=CONV_CHAINS)
        # simulation study
        #expand(SCORE_DIR+"/mcmc_d={d}/v={v}_r={r}_a={a}_t={t}.json", 
        #       d=REG_DEGS, v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]), 
        #expand(SCORE_DIR+"/funchisq/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/hill/v={v}_r={r}_a={a}_t={t}.json",  
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand("simulation-study/scores/lasso/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        ## Hill timetest results
        #FIG_DIR+"/hill_method_timetest.csv"    

rule simulate_data:
    input:
        simulator="builddir/simulate_data"
    output:
        ts=TS_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        ref=REF_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        true=TRU_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv"
    shell:
        "{input.simulator} {wildcards.v} {wildcards.t} {SIM_M} {wildcards.r} {wildcards.a} {POLY_DEG} {output.ref} {output.true} {output.ts}"

rule score_predictions:
    input:
        "builddir/scoring",
        tr_dg_fs=[TRU_DIR+"/{sim_gridpoint}_replicate="+str(rep)+".csv" for rep in SIM_REPLICATES],
        pp_res=[PRED_DIR+"/{method}/{sim_gridpoint}_replicate="+str(rep)+".json" for rep in SIM_REPLICATES]
    output:
        out=SCORE_DIR+"/{method}/{sim_gridpoint}.json" 
    shell:
        "builddir/scoring --truth-files {input.tr_dg_fs} --pred-files {input.pp_res} --output-file {output.out}"



######################
# MCMC JOBS

rule run_conv_mcmc_sim:
    input:
        method=BIN_DIR+"/mcmc/Catsupp",
        ts_file=TS_DIR+"/{dataset}.csv",
        ref_dg=REF_DIR+"/{dataset}.csv",
    output:
        CONV_RAW_DIR+"/{dataset}/mcmc_d={d}/{chain}.bson"
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} {output} {CONV_TIMEOUT}"\
        +" --regression-deg {wildcards.d} --n-samples {CONV_MAX_SAMPLES}"

rule postprocess_sim_mcmc:
    input:
        "builddir/postprocess",
        raw=RAW_DIR+"/mcmc_{mcmc_settings}/{replicate}.json" 
    output:
        out=PRED_DIR+"/mcmc_{mcmc_settings}/{replicate}.json"
    shell:
        "builddir/postprocess {input.raw} --output-file {output.out}"

rule run_sim_mcmc:
    input:
        method=BIN_DIR+"/mcmc/Catsupp",
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv",
    output:
        RAW_DIR+"/mcmc_d={d}/{replicate}.json"
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} {output} {SIM_TIMEOUT}"\
        +" --regression-deg {wildcards.d}"
        

# END MCMC JOBS
#####################

#####################
# FUNCHISQ JOBS

rule run_sim_funchisq:
    input: 
        FUNCH_DIR+"/funchisq_wrapper.R",
        ts_file=TS_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/funchisq/{replicate}.json"
    shell:
        "Rscript {FUNCH_DIR}/funchisq_wrapper.R {input.ts_file} {output}"

# END FUNCHISQ JOBS
#####################

#####################
# HILL JOBS

rule run_sim_hill:
    input:
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/hill_{deg}_{mode}/{replicate}.json"
    shell:
        "matlab -nodesktop -nosplash -nojvm -r \'cd(\"{HILL_DIR}\"); try, hill_dbn_wrapper(\"{input.ts_file}\", \"{input.ref_dg}\", \"{output}\", -1, \"full\", {SIM_TIMEOUT}), catch e, quit(1), end, quit\'"


rule run_timetest_hill:
    input:
        ts=TS_DIR+"/{replicate}.csv",
        ref=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/hill_{deg}_{mode}/{replicate}.json"
    shell:
        "matlab -nodesktop -nosplash -nojvm -r \'cd(\""+HILL_DIR+"\"); try, hill_dbn_wrapper(\"{input.ts}\", \"{input.ref}\", \"{output}\", {wildcards.deg}, \"{wildcards.mode}\", "+str(HILL_TIME_TIMEOUT)+"), catch e, quit(1), end, quit\'"


rule tabulate_timetest_hill:
    input:
        [PRED_DIR+"/hill_"+str(comb[0])+"_"+str(m)+"/v="+str(comb[1])+"_r="+str(SIM_GRID["R"][0])+"_a="+str(SIM_GRID["A"][0])+"_t="+str(SIM_GRID["T"][0])+"_replicate=0.json" for comb in HILL_TIME_COMBS for m in HILL_MODES]
    output:
        FIG_DIR+"/hill_method_timetest.csv"
    script:
        HILL_DIR+"/tabulate_timetest_results.py"

# END HILL JOBS
#####################

########################
# LASSO JOBS
rule score_sim_lasso:
    input:
        ""
    output:
        ""
    shell:
        ""

rule run_sim_lasso:
    input:
        ""
    output:
        ""
    shell:
        ""

# END LASSO JOBS
########################

########################
# JULIA CODE COMPILATION

# Get the path of the Julia PackageCompiler
JULIAC_PATH = glob.glob(os.path.join(os.environ["HOME"],
          ".julia/packages/PackageCompiler/*/juliac.jl")
          )[0]

rule compile_simulator:
    input:
        "simulation-study/simulate_data.jl"
    output:
        BIN_DIR+"/simulator/simulate_data"
    params:
        simulator_dir=BIN_DIR+"/simulator"
    shell:
        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -d {params.simulator_dir} -vaet simulation-study/simulate_data.jl"

rule compile_postprocessor:
    input:
        BIN_DIR+"/mcmc/Catsupp",
        JULIA_PROJ_DIR+"/postprocess.jl"
    output:
        "builddir/postprocess"
    shell:
        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -vaet {JULIA_PROJ_DIR}/postprocess.jl"


rule compile_scoring:
    input:
        JULIA_PROJ_DIR+"/scoring.jl",
        "builddir/postprocess"
    output:
        "builddir/scoring"
    shell:
        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -vaet {JULIA_PROJ_DIR}/scoring.jl"

rule compile_mcmc:
    input:
        "builddir/simulate_data",
        JULIA_PROJ_DIR+"/Catsupp.jl"
    output:
        BIN_DIR+"/mcmc/Catsupp"
    params:
        mcmc_bin=BIN_DIR+"/mcmc"
    shell:
        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -d {params.mcmc_bin} -vaet {JULIA_PROJ_DIR}/Catsupp.jl"

rule compile_lasso:
    input:
        JULIA_PROJ_DIR+"/lasso.jl",
        "builddir/simulate_data"
    output:
        "builddir/lasso"
    shell:
        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -vaet {JULIA_PROJ_DIR}/lasso.jl"

# END JULIA CODE COMPILATION
#############################

        
