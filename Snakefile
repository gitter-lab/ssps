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
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")
JULIA_PROJ_DIR = os.path.join(ROOT_DIR, "julia-project")
HILL_DIR = os.path.join(ROOT_DIR, "hill-method")
FUNCH_DIR = os.path.join(ROOT_DIR, "funchisq")
TEMP_DIR = config["temp_dir"]
EXP_DATA_DIR = os.path.join(ROOT_DIR,"experimental_data")
HILL_TIME_DIR = os.path.join(TEMP_DIR, "time_tests")

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
SIM_MAX_SAMPLES = MC_PARAMS["max_samples"]
REG_DEGS = MC_PARAMS["regression_deg"]
BURNIN = MC_PARAMS["burnin"]
SIM_CHAINS=list(range(MC_PARAMS["n_chains"]))

# Hill hyperparameters
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
CONV_BURNIN = CONV_PARAMS["burnin"]
CONV_STOPPOINTS = CONV_PARAMS["stop_points"]
CONV_NEFF = CONV_PARAMS["neff_per_chain"] * len(CONV_CHAINS)
CONV_PSRF = CONV_PARAMS["psrf_ub"]


#############################
# RULES
#############################
rule all:
    input:
        # convergence tests on simulated data
        #expand(FIG_DIR+"/convergence/v={v}_r={r}_a={a}_t={t}_d={d}.png",
        #       v=CONV_SIM_GRID["V"], r=CONV_SIM_GRID["R"], a=CONV_SIM_GRID["A"],
        #       t=CONV_SIM_GRID["T"], d=CONV_DEGS)
        ## convergence tests on experimental data
        #expand(CONV_RAW_DIR+"/{dataset}/mcmc_d={d}/chain_{c}.json", 
        #       ds=CONV_DATASETS, d=CONV_DEGS, c=CONV_CHAINS)
        # MCMC simulation scores
        #expand(FIG_DIR+"/simulation_study/mcmc_d={d}/v={v}_t={t}.csv", 
        #       d=REG_DEGS, v=SIM_GRID["V"], t=SIM_GRID["T"])
        #expand(SCORE_DIR+"/funchisq/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        expand(SCORE_DIR+"/hill/v={v}_r={r}_a={a}_t={t}.json",  
               v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand("simulation-study/scores/lasso/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/prior_baseline/v={v}_r={r}_a={a}_t={t}.json",  
        #       v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        # Hill timetest results
        #FIG_DIR+"/hill_method_timetest.csv"    

rule simulate_data:
    input:
        simulator=BIN_DIR+"/simulate_data/simulate_data"
    output:
        ts=TS_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        ref=REF_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        true=TRU_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv"
    shell:
        "{input.simulator} {wildcards.v} {wildcards.t} {SIM_M} {wildcards.r} {wildcards.a} {POLY_DEG} {output.ref} {output.true} {output.ts}"

rule score_predictions:
    input:
        scorer=BIN_DIR+"/scoring/scoring",
        tr_dg_fs=[TRU_DIR+"/{sim_gridpoint}_replicate="+str(rep)+".csv" for rep in SIM_REPLICATES],
        pp_res=[PRED_DIR+"/{method}/{sim_gridpoint}_replicate="+str(rep)+".json" for rep in SIM_REPLICATES]
    output:
        out=SCORE_DIR+"/{method}/{sim_gridpoint}.json" 
    shell:
        "{input.scorer} --truth-files {input.tr_dg_fs} --pred-files {input.pp_res} --output-file {output.out}"


##########################
# VISUALIZATION RULES

rule convergence_viz:
    input:
        expand(CONV_RES_DIR+"/v={{v}}_r={{r}}_a={{a}}_t={{t}}_replicate={rep}/mcmc_d={{d}}.json",
               rep=CONV_REPLICATES) 
    output:
        FIG_DIR+"/convergence/v={v}_r={r}_a={a}_t={t}_d={d}.png"
    script:
        SCRIPT_DIR+"/convergence_viz.py"

rule sim_study_score_viz:
    input:
        expand(SCORE_DIR+"/{{method}}/v={{v}}_r={r}_a={a}_t={{t}}.json", 
               r=SIM_GRID["R"], a=SIM_GRID["A"]), 
    output:
        FIG_DIR+"/simulation_study/{method}/v={v}_t={t}.csv"
    script:
        SCRIPT_DIR+"/sim_study_score_viz.py"

######################
# MCMC JOBS

rule postprocess_conv_mcmc_sim:
    input:
        pp=BIN_DIR+"/postprocess_samples/postprocess_samples",
        raw=expand(CONV_RAW_DIR+"/{{dataset}}/{{mcmc_settings}}/{chain}.json", chain=CONV_CHAINS)
    output:
        out=CONV_RES_DIR+"/{dataset}/{mcmc_settings}.json"
    shell:
        "{input.pp} --chain-samples {input.raw} --output-file {output.out} --burnin {CONV_BURNIN}"
        +" --stop-points {CONV_STOPPOINTS}" 

rule run_conv_mcmc_sim:
    input:
        method=BIN_DIR+"/Catsupp/Catsupp",
        ts_file=TS_DIR+"/{dataset}.csv",
        ref_dg=REF_DIR+"/{dataset}.csv",
    output:
        CONV_RAW_DIR+"/{dataset}/mcmc_d={d}/{chain}.json"
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} {output} {CONV_TIMEOUT}"\
        +" --store-samples --n-steps {CONV_MAX_SAMPLES} --regression-deg {wildcards.d}"

rule postprocess_sim_mcmc:
    input:
        pp=BIN_DIR+"/postprocess_counts/postprocess_counts",
        raw=expand(RAW_DIR+"/mcmc_{{mcmc_settings}}/{{replicate}}/{chain}.json",
                   chain=SIM_CHAINS)
    output:
        out=PRED_DIR+"/mcmc_{mcmc_settings}/{replicate}.json"
    shell:
        "{input.pp} {input.raw} --output-file {output.out}"

rule run_sim_mcmc:
    input:
        method=BIN_DIR+"/Catsupp/Catsupp",
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv",
    output:
        RAW_DIR+"/mcmc_d={d}/{replicate}/{chain}.json"
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} {output} {SIM_TIMEOUT}"\
        +" --regression-deg {wildcards.d} --n-steps {SIM_MAX_SAMPLES}"
        

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
        PRED_DIR+"/hill/{replicate}.json"
    shell:
        "matlab -nodesktop -nosplash -nojvm -r \'cd(\"{HILL_DIR}\"); try, hill_dbn_wrapper(\"{input.ts_file}\", \"{input.ref_dg}\", \"{output}\", -1, \"auto\", {SIM_TIMEOUT}), catch e, quit(1), end, quit\'"


rule run_timetest_hill:
    input:
        ts=TS_DIR+"/{replicate}.csv",
        ref=REF_DIR+"/{replicate}.csv"
    output:
        HILL_TIME_DIR+"/preds/hill_deg={deg}_mode={mode}/{replicate}.json"
    shell:
        "matlab -nodesktop -nosplash -nojvm -r \'cd(\""+HILL_DIR+"\"); try, hill_dbn_wrapper(\"{input.ts}\", \"{input.ref}\", \"{output}\", {wildcards.deg}, \"{wildcards.mode}\", "+str(HILL_TIME_TIMEOUT)+"), catch e, quit(1), end, quit\'"


rule tabulate_timetest_hill:
    input:
        [HILL_TIME_DIR+"/preds/hill_deg="+str(comb[0])+"_mode="+str(m)+"/v="+str(comb[1])+"_r="+str(SIM_GRID["R"][0])+"_a="+str(SIM_GRID["A"][0])+"_t="+str(SIM_GRID["T"][0])+"_replicate=0.json" for comb in HILL_TIME_COMBS for m in HILL_MODES]
    output:
        FIG_DIR+"/hill_method_timetest.csv"
    script:
        SCRIPT_DIR+"/tabulate_timetest_results.py"

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
# BASELINE JOBS 
rule run_sim_prior_baseline:
    input:
        method=BIN_DIR+"/prior_baseline/prior_baseline",
        ref=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/prior_baseline/{replicate}.json"
    shell:
        "{input.method} {input.ref} {output}"

# END BASELINE JOBS
#######################

########################
# JULIA CODE COMPILATION

# Get the path of the Julia PackageCompiler
JULIAC_PATH = glob.glob(os.path.join(os.environ["HOME"],
          ".julia/packages/PackageCompiler/*/juliac.jl")
          )[0]

rule compile_julia:
    input:
        src=JULIA_PROJ_DIR+"/{source_name}.jl"
    output:
        exe=BIN_DIR+"/{source_name}/{source_name}"
    params:
        outdir=BIN_DIR+"/{source_name}"
    threads: 2 
    shell:
        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -d {params.outdir} -vaet {input.src}"


# END JULIA CODE COMPILATION
#############################

        
