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
DREAM_DIR = os.path.join(ROOT_DIR, "dream-challenge")
DREAM_TS_DIR = os.path.join(DREAM_DIR, "train")
DREAM_REF_DIR = os.path.join(DREAM_DIR, "prior")
DREAM_TRU_DIR = os.path.join(DREAM_DIR, "test")
TEMP_DIR = config["temp_dir"]
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
LAMBDA_PROP_STD = MC_PARAMS["lambda_prop_std"]

# Experimental evaluation directories
EXP_DIR = os.path.join(TEMP_DIR,"experimental_eval")
EXPDAT_DIR = os.path.join(EXP_DIR, "datasets")
EXP_TS_DIR = os.path.join(EXPDAT_DIR, "timeseries")
EXP_REF_DIR = os.path.join(EXPDAT_DIR, "ref_graphs")
EXP_RAW_DIR = os.path.join(EXP_DIR, "raw")
EXP_PRED_DIR = os.path.join(EXP_DIR, "predictions")
EXP_SCORE_DIR = os.path.join(EXP_DIR, "scores")

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
CONV_DEGS = CONV_PARAMS["mcmc_hyperparams"]["regression_deg"]
CONV_LAMBDA_STDS = CONV_PARAMS["mcmc_hyperparams"]["lambda_prop_stds"]
CONV_REPLICATES = list(range(CONV_PARAMS["N"]))
CONV_CHAINS = list(range(CONV_PARAMS["n_chains"]))
CONV_MAX_SAMPLES = CONV_PARAMS["max_samples"]
CONV_TIMEOUT = CONV_PARAMS["timeout"]
CONV_BURNIN = CONV_PARAMS["burnin"]
CONV_STOPPOINTS = CONV_PARAMS["stop_points"]
CONV_NEFF = CONV_PARAMS["neff_per_chain"] * len(CONV_CHAINS)
CONV_PSRF = CONV_PARAMS["psrf_ub"]
EXP_CELL_LINES = CONV_PARAMS["experiments"]["cell_lines"]
STIMULI = CONV_PARAMS["experiments"]["stimuli"]


#############################
# RULES
#############################
rule all:
    input:
        # Convergence tests on simulated data
        #expand(FIG_DIR+"/convergence/v={v}_r={r}_a={a}_t={t}_d={d}.png",
        #       v=CONV_SIM_GRID["V"], r=CONV_SIM_GRID["R"], a=CONV_SIM_GRID["A"],
        #       t=CONV_SIM_GRID["T"], d=CONV_DEGS)
        # Convergence tests on experimental data
        expand(FIG_DIR+"/convergence/cl={cell_line}_stim={stimulus}_d={d}_lstd={lstd}.png", 
               cell_line=EXP_CELL_LINES, stimulus=STIMULI, d=CONV_DEGS, lstd=CONV_LAMBDA_STDS),
        # Simulation scores
        #expand(FIG_DIR+"/simulation_study/mcmc_d={d}/v={v}_t={t}.csv", 
        #       d=REG_DEGS, v=SIM_GRID["V"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/funchisq/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/hill/v={v}_r={r}_a={a}_t={t}.json",  
        #       v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/lasso/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/prior_baseline/v={v}_r={r}_a={a}_t={t}.json",  
        #       v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        # DREAM scores
        #expand(EXP_SCORE_DIR+"/mcmc_d={d}/cl={cell_line}_stim={stimulus}.json", 
        #       d=REG_DEGS, cell_line=EXP_CELL_LINES, stimulus=STIMULI),
        #expand(EXP_SCORE_DIR+"/funchisq/cl={cell_line}_stim={stimulus}.json", 
        #       cell_line=EXP_CELL_LINES, stimulus=STIMULI),
        #expand(EXP_SCORE_DIR+"/hill/cl={cell_line}_stim={stimulus}.json", 
        #       cell_line=EXP_CELL_LINES, stimulus=STIMULI)
        #expand(EXP_SCORE_DIR+"/prior_baseline/cl={cell_line}_stim={stimulus}.json", 
        #       cell_line=EXP_CELL_LINES, stimulus=STIMULI)
        # Hill timetest results
        #FIG_DIR+"/hill_method_timetest.csv"    


rule simulate_data:
    input:
        simulator=JULIA_PROJ_DIR+"/simulate_data.jl"
    output:
        ts=TS_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        ref=REF_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        true=TRU_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv"
    resources:
        mem_mb=10,
        threads=1
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.simulator} {wildcards.v} {wildcards.t} {SIM_M} {wildcards.r} {wildcards.a} {POLY_DEG} {output.ref} {output.true} {output.ts}"


rule score_sim_predictions:
    input:
        scorer=JULIA_PROJ_DIR+"/scoring.jl",
        tr_dg_fs=[TRU_DIR+"/{sim_gridpoint}_replicate="+str(rep)+".csv" for rep in SIM_REPLICATES],
        pp_res=[PRED_DIR+"/{method}/{sim_gridpoint}_replicate="+str(rep)+".json" for rep in SIM_REPLICATES]
    output:
        out=SCORE_DIR+"/{method}/{sim_gridpoint}.json" 
    resources:
        mem_mb=100,
        threads=1
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.scorer} --truth-files {input.tr_dg_fs} --pred-files {input.pp_res} --output-file {output.out}"


##########################
# VISUALIZATION RULES

rule convergence_viz:
    input:
        expand(CONV_RES_DIR+"/{{dataset}}_replicate={rep}/mcmc_d={{d}}_lstd={{lstd}}.json",
               rep=CONV_REPLICATES) 
    output:
        FIG_DIR+"/convergence/{dataset}_d={d}_lstd={lstd}.png"
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
        pp=JULIA_PROJ_DIR+"/postprocess_samples.jl",
        raw=expand(CONV_RAW_DIR+"/{{dataset}}/{{mcmc_settings}}/{chain}.json", chain=CONV_CHAINS)
    output:
        out=CONV_RES_DIR+"/{dataset}/{mcmc_settings}.json"
    resources:
        runtime=3600,
        threads=1,
        mem_mb=6000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.pp} --chain-samples {input.raw} --output-file {output.out} --burnin {CONV_BURNIN}"
        +" --stop-points {CONV_STOPPOINTS}" 

rule run_conv_mcmc_sim:
    input:
        method=JULIA_PROJ_DIR+"/Catsupp.jl",
        ts_file=TS_DIR+"/{dataset}.csv",
        ref_dg=REF_DIR+"/{dataset}.csv",
    output:
        CONV_RAW_DIR+"/{dataset}/mcmc_d={d}/{chain}.json"
    resources:
        runtime=int(CONV_TIMEOUT)+60,
        threads=1,
        mem_mb=4000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {CONV_TIMEOUT}"\
        +" --store-samples --n-steps {CONV_MAX_SAMPLES} --regression-deg {wildcards.d}"\
        +" --lambda-prop-std 3.0"

rule postprocess_sim_mcmc:
    input:
        pp=JULIA_PROJ_DIR+"/postprocess_counts.jl",
        raw=expand(RAW_DIR+"/mcmc_{{mcmc_settings}}/{{replicate}}/{chain}.json",
                   chain=SIM_CHAINS)
    output:
        out=PRED_DIR+"/mcmc_{mcmc_settings}/{replicate}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=2000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.pp} {input.raw} --output-file {output.out}"

rule run_sim_mcmc:
    input:
        method=JULIA_PROJ_DIR+"/Catsupp.jl",
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv",
    output:
        RAW_DIR+"/mcmc_d={d}/{replicate}/{chain}.json"
    resources:
        runtime=SIM_TIMEOUT,
        threads=1,
        mem_mb=2000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {SIM_TIMEOUT}"\
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
    resources:
        runtime=60,
        threads=1,
        mem_mb=500
    shell:
        "Rscript {FUNCH_DIR}/funchisq_wrapper.R {input.ts_file} {output}"

# END FUNCHISQ JOBS
#####################

#####################
# HILL JOBS

def hill_mem(wildcards):
    v = int(wildcards.replicate.split("_")[0].split("=")[1])
    return int(8000 * v / 500)

rule run_sim_hill:
    input:
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/hill/{replicate}.json"
    resources:
        runtime=SIM_TIMEOUT+60,
        threads=1,
        mem_mb=hill_mem
    shell:
        "matlab -nodesktop -nosplash -nojvm -singleCompThread -r \'cd(\"{HILL_DIR}\"); try, hill_dbn_wrapper(\"{input.ts_file}\", \"{input.ref_dg}\", \"{output}\", -1, \"auto\", {SIM_TIMEOUT}), catch e, quit(1), end, quit\'"


rule run_timetest_hill:
    input:
        ts=TS_DIR+"/{replicate}.csv",
        ref=REF_DIR+"/{replicate}.csv"
    output:
        HILL_TIME_DIR+"/preds/hill_deg={deg}_mode={mode}/{replicate}.json"
    shell:
        "matlab -nodesktop -nosplash -nojvm -singleCompThread -r \'cd(\""+HILL_DIR+"\"); try, hill_dbn_wrapper(\"{input.ts}\", \"{input.ref}\", \"{output}\", {wildcards.deg}, \"{wildcards.mode}\", "+str(HILL_TIME_TIMEOUT)+"), catch e, quit(1), end, quit\'"


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
        method=JULIA_PROJ_DIR+"/prior_baseline.jl",
        ref=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/prior_baseline/{replicate}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=100
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ref} {output}"

# END BASELINE JOBS
#######################


##############################
# EXPERIMENTAL DATA JOBS

rule score_dream_predictions:
    input:
        scorer=SCRIPT_DIR+"/score_dream.py",
        tr_desc=DREAM_TRU_DIR+"/TrueVec_{cell_line}_{stim}.csv", 
        preds=EXP_PRED_DIR+"/{method}/cl={cell_line}_stim={stim}.json",
        ab=EXP_REF_DIR+"/cl={cell_line}_antibodies.json",
    output:
        out=EXP_SCORE_DIR+"/{method}/cl={cell_line}_stim={stim}.json" 
    resources:
        mem_mb=100,
        threads=1
    shell:
        "python {input.scorer} {input.preds} {input.tr_desc} {input.ab} {output.out}"


rule postprocess_dream_mcmc:
    input:
        pp=JULIA_PROJ_DIR+"/postprocess_counts.jl",
        raw=expand(EXP_RAW_DIR+"/mcmc_{{mcmc_settings}}/{{context}}/chain={chain}.json",
                   chain=SIM_CHAINS)
    output:
        out=EXP_PRED_DIR+"/mcmc_{mcmc_settings}/{context}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=2000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.pp} {input.raw} --output-file {output.out}"


rule run_dream_conv:
    input:
        method=JULIA_PROJ_DIR+"/Catsupp.jl",
        ts_file=EXP_TS_DIR+"/cl={cell_line}_stim={stimulus}.csv",
        ref_dg=EXP_REF_DIR+"/cl={cell_line}.csv",
    output:
        CONV_RAW_DIR+"/cl={cell_line}_stim={stimulus}_replicate={replicate}/mcmc_d={d}_lstd={lstd}/{chain}.json"
    resources:
        runtime=3600,
        threads=1,
        mem_mb=3000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {CONV_TIMEOUT}"\
        +" --store-samples --n-steps {CONV_MAX_SAMPLES} --regression-deg {wildcards.d}"\
        +" --continuous-reference --lambda-prop-std {wildcards.lstd}"


rule run_dream_mcmc:
    input:
        method=JULIA_PROJ_DIR+"/Catsupp.jl",
        ts_file=EXP_TS_DIR+"/cl={cell_line}_stim={stim}.csv",
        ref_dg=EXP_REF_DIR+"/cl={cell_line}.csv",
    output:
        EXP_RAW_DIR+"/mcmc_d={d}/cl={cell_line}_stim={stim}/chain={chain}.json"
    resources:
        runtime=SIM_TIMEOUT,
        threads=1,
        mem_mb=2000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {SIM_TIMEOUT}"\
        +" --regression-deg {wildcards.d} --n-steps {SIM_MAX_SAMPLES} --continuous-reference"


rule run_dream_funchisq:
    input: 
        method=FUNCH_DIR+"/funchisq_wrapper.R",
        ts_file=EXP_TS_DIR+"/{replicate}.csv"
    output:
        EXP_PRED_DIR+"/funchisq/{replicate}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=500
    shell:
        "Rscript {input.method} {input.ts_file} {output}"


rule run_dream_hill:
    input:
        ts_file=EXP_TS_DIR+"/cl={cell_line}_stim={stim}.csv",
        ref_dg=EXP_REF_DIR+"/cl={cell_line}.csv"
    output:
        EXP_PRED_DIR+"/hill/cl={cell_line}_stim={stim}.json"
    resources:
        runtime=SIM_TIMEOUT+60,
        threads=1,
        mem_mb=2000
    shell:
        "matlab -nodesktop -nosplash -nojvm -singleCompThread -r \'cd(\"{HILL_DIR}\"); try, hill_dbn_wrapper(\"{input.ts_file}\", \"{input.ref_dg}\", \"{output}\", -1, \"auto\", {SIM_TIMEOUT}), catch e, quit(1), end, quit\'"


rule run_dream_prior_baseline:
    input:
        method=JULIA_PROJ_DIR+"/prior_baseline.jl",
        ref=EXP_REF_DIR+"/cl={cell_line}.csv"
    output:
        EXP_PRED_DIR+"/prior_baseline/cl={cell_line}_stim={stim}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=100
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ref} {output}"


rule preprocess_dream_timeseries:
    input:
        DREAM_TS_DIR+"/{cell_line}_main.csv"
    output:
        expand(EXP_TS_DIR+"/cl={{cell_line}}_stim={stimulus}.csv", stimulus=STIMULI)
    shell:
        "python scripts/preprocess_dream_ts.py {input} {EXP_TS_DIR} --ignore-inhibitor"


rule preprocess_dream_prior:
    input:
        DREAM_REF_DIR+"/{cell_line}.eda"
    output:
        edges=EXP_REF_DIR+"/cl={cell_line}.csv",
        ab=EXP_REF_DIR+"/cl={cell_line}_antibodies.json"
    shell:
        "python scripts/preprocess_dream_prior.py {input} {output.edges} {output.ab}"

# END EXPERIMENTAL DATA JOBS
##############################


########################
# JULIA CODE COMPILATION

# Get the path of the Julia PackageCompiler
#JULIAC_PATH = glob.glob(os.path.join(os.environ["HOME"],
#          ".julia/packages/PackageCompiler/*/juliac.jl")
#          )[0]
#
#rule compile_julia:
#    input:
#        src=JULIA_PROJ_DIR+"/{source_name}.jl"
#    output:
#        exe=BIN_DIR+"/{source_name}/{source_name}"
#    params:
#        outdir=BIN_DIR+"/{source_name}",
#    resources:
#        threads=1,
#        mem_mb=2000
#    shell:
#        "julia --project={JULIA_PROJ_DIR} {JULIAC_PATH} -d {params.outdir} -vaet {input.src}"


# END JULIA CODE COMPILATION
#############################

        
