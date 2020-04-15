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
DREAM_RAW_DIR = os.path.join(ROOT_DIR, "dream-challenge")
DREAM_TS_DIR = os.path.join(DREAM_RAW_DIR, "train")
DREAM_PRIOR_DIR = os.path.join(DREAM_RAW_DIR, "prior")
DREAM_TRU_DIR = os.path.join(DREAM_RAW_DIR, "test")

TEMP_DIR = config["temp_dir"]
if TEMP_DIR == "": # Default location of temp directory
    TEMP_DIR = os.path.join(ROOT_DIR, "temp")

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
SIM_TIMEOUT = SIM_PARAMS["prediction"]["timeout"]
SIM_REPLICATES = list(range(SIM_PARAMS["prediction"]["N"]))
SIM_GRID = SIM_PARAMS["simulation_grid"]
SIM_M = SIM_GRID["M"]
POLY_DEG = SIM_PARAMS["polynomial_degree"]
SIM_BASELINES = SIM_PARAMS["baseline_methods"]

# MCMC hyperparameters (for simulation study)
MC_PARAMS = SIM_PARAMS["mcmc_hyperparams"]
SIM_MAX_SAMPLES = MC_PARAMS["max_samples"]
REG_DEGS = MC_PARAMS["regression_deg"]
BURNIN = MC_PARAMS["burnin"]
SIM_CHAINS=list(range(SIM_PARAMS["prediction"]["n_chains"]))
LAMBDA_PROP_STD = MC_PARAMS["lambda_prop_std"]

# Produce a list of all methods used in the simulation study
SIM_METHODS = SIM_BASELINES + ["mcmc_d={}".format(deg) for deg in REG_DEGS]

# Hill hyperparameters
HILL_TIME_PARAMS = config["hill_timetest"]
HILL_TIME_COMBS = HILL_TIME_PARAMS["deg_v_combs"]
HILL_MODES = HILL_TIME_PARAMS["modes"]
HILL_TIME_TIMEOUT = HILL_TIME_PARAMS["timeout"]

# Convergence analysis 
CONV_DIR = os.path.join(TEMP_DIR, "convergence")
CONV_RES_DIR = os.path.join(CONV_DIR, "results")
CONV_RAW_DIR = os.path.join(CONV_DIR, "raw")
CONV_PARAMS = config["simulation_study"]["convergence"]
CONV_DEGS = MC_PARAMS["regression_deg"]
CONV_LAMBDA_STDS = MC_PARAMS["lambda_prop_std"]
MCMC_INDEG = MC_PARAMS["large_indeg"]
CONV_REPLICATES = list(range(CONV_PARAMS["N"]))
CONV_CHAINS = list(range(CONV_PARAMS["n_chains"]))
CONV_MAX_SAMPLES = MC_PARAMS["max_samples"]
CONV_TIMEOUT = CONV_PARAMS["timeout"]
CONV_BURNIN = MC_PARAMS["burnin"]
CONV_STOPPOINTS = CONV_PARAMS["stop_points"]
CONV_NEFF = CONV_PARAMS["neff_per_chain"] * len(CONV_CHAINS)
CONV_PSRF = CONV_PARAMS["psrf_ub"]

# Experimental evaluation directories
DREAM_DIR = os.path.join(TEMP_DIR,"experimental_eval")
DREAM_DAT_DIR = os.path.join(DREAM_DIR, "datasets")
DREAM_PREP_TS_DIR = os.path.join(DREAM_DAT_DIR, "timeseries")
DREAM_REF_DIR = os.path.join(DREAM_DAT_DIR, "ref_graphs")
DREAM_OUT_DIR = os.path.join(DREAM_DIR, "raw_output")
DREAM_PRED_DIR = os.path.join(DREAM_DIR, "predictions")
DREAM_SCORE_DIR = os.path.join(DREAM_DIR, "scores")

# Experimental evaluation parameters
DREAM_PARAMS = config["dream_challenge"]
CELL_LINES = DREAM_PARAMS["cell_lines"]
STIMULI = DREAM_PARAMS["stimuli"]
DREAM_REPLICATES = list(range(DREAM_PARAMS["N"]))
DREAM_CONV_PARAMS = DREAM_PARAMS["convergence"]
DREAM_STOPPOINTS = DREAM_CONV_PARAMS["stop_points"]
DREAM_NEFF = DREAM_CONV_PARAMS["neff_per_chain"]
DREAM_PSRF = DREAM_CONV_PARAMS["psrf_ub"]
DREAM_MC_PARAMS = DREAM_PARAMS["mcmc_hyperparams"]
DREAM_TIMEOUT = DREAM_MC_PARAMS["timeout"]
DREAM_MAX_SAMPLES = DREAM_MC_PARAMS["max_samples"]
DREAM_LSTD = DREAM_MC_PARAMS["lambda_prop_std"]
DREAM_CHAINS = list(range(DREAM_MC_PARAMS["n_chains"]))
DREAM_REGDEGS = DREAM_MC_PARAMS["regression_deg"]
DREAM_LARGE_INDEG = DREAM_MC_PARAMS["large_indeg"]
DREAM_BASELINES = DREAM_PARAMS["baseline_methods"] 

DREAM_METHODS = DREAM_BASELINES[:]
DREAM_METHODS += ["mcmc_d={}_lstd={}".format(deg, lstd) for deg in DREAM_REGDEGS for lstd in DREAM_LSTD]

#############################
# RULES
#############################
rule all:
    input:
        # Convergence tests on simulated data
        #expand(FIG_DIR+"/convergence/v={v}_r={r}_a={a}_t={t}_d={d}.png",
        #       v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"],
        #       t=SIM_GRID["T"], d=CONV_DEGS)
        # Convergence tests on experimental data
        #expand(FIG_DIR+"/dream/convergence/mcmc_d={d}_lstd={lstd}/cl={cell_line}_stim={stimulus}.png", 
        #       cell_line=CELL_LINES, stimulus=STIMULI, d=DREAM_REGDEGS, lstd=DREAM_LSTD),
        #expand(FIG_DIR+"/simulation_study/heatmaps/{method}-mcmc_d=1-{score}-{style}.png",
        #        method=SIM_BASELINES, score=["aucroc","aucpr"], style=["mean","t"])
        # Simulation scores
        SIM_DIR+"/sim_scores.tsv",
        # DREAM scores
        #DREAM_DIR+"/dream_scores.tsv"
        # Hill timetest results
        #FIG_DIR+"/hill_method_timetest.csv"


rule sim_heatmaps:
    input:
        table=SIM_DIR+"/sim_scores.tsv",
    output:
        heatmaps=FIG_DIR+"/simulation_study/heatmaps/{method1}-{method2}-{score}.png"
    shell:
        "python scripts/sim_heatmap.py {input.table} {output.heatmaps} {wildcards.score} {wildcards.method1} {wildcards.method2}"


rule tabulate_sim_scores:
    input:
        scores=expand(SCORE_DIR+"/{method}/v={v}_r={r}_a={a}_t={t}_replicate={rep}.json",
                      method=SIM_METHODS, v=SIM_GRID["V"], r=SIM_GRID["R"], 
                      a=SIM_GRID["A"], t=SIM_GRID["T"], rep=SIM_REPLICATES),
    output:
        SIM_DIR+"/sim_scores.tsv"
    shell:
        "python scripts/tabulate_scores.py {input.scores} {output}"


rule tabulate_dream_scores:
    input:
        mcmc=expand(DREAM_SCORE_DIR+"/mcmc_d={d}_lstd={lstd}/cl={cell_line}_stim={stimulus}_replicate={rep}.json", 
                    d=REG_DEGS, cell_line=CELL_LINES, stimulus=STIMULI, lstd=DREAM_LSTD, rep=DREAM_REPLICATES),
        baselines=expand(DREAM_SCORE_DIR+"/{method}/cl={cell_line}_stim={stimulus}.json",
                         method=DREAM_BASELINES, cell_line=CELL_LINES, stimulus=STIMULI)
    output:
        DREAM_DIR+"/dream_scores.tsv"
    shell:
        "python scripts/tabulate_scores.py {input.mcmc} {input.baselines}"


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
        tr_dg=TRU_DIR+"/{replicate}.csv",
        pp_res=PRED_DIR+"/{method}/{replicate}.json"
    output:
        out=SCORE_DIR+"/{method}/{replicate}.json" 
    resources:
        mem_mb=100,
        threads=1
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.scorer} --truth-file {input.tr_dg} --pred-file {input.pp_res} --output-file {output.out}"


##########################
# SIM VISUALIZATION RULES


#rule sim_heatmaps:
#    input:
#        SIM_DIR+"/sim_scores.tsv"
#    output:
#        mean=FIG_DIR+"/simulation_study/heatmaps/{method}-{mcmc_method}-{score}-mean.png",
#        t=FIG_DIR+"/simulation_study/heatmaps/{method}-{mcmc_method}-{score}-t.png"
#    shell:
#        "python scripts/sim_heatmap.py {input} {output.mean} {output.t} {wildcards.score} prior_baseline {wildcards.method} {wildcards.mcmc_method}" 

rule convergence_viz:
    input:
        expand(PRED_DIR+"/mcmc_d={{d}}_lstd={{lstd}}/{{dataset}}_replicate={rep}.json",
               rep=CONV_REPLICATES) 
    output:
        FIG_DIR+"/simulation_study/convergence/mcmc_d={d}_lstd={lstd}/{dataset}.png"
    script:
        SCRIPT_DIR+"/convergence_viz.py"



######################
# MCMC JOBS

rule postprocess_conv_mcmc_sim:
    input:
        pp=JULIA_PROJ_DIR+"/postprocess_samples.jl",
        raw=expand(CONV_RAW_DIR+"/{{mcmc_settings}}/{{dataset}}/{chain}.json", chain=CONV_CHAINS)
    output:
        out=CONV_RES_DIR+"/{mcmc_settings}/{dataset}.json"
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
        CONV_RAW_DIR+"/mcmc_d={d}_lstd={lstd}/{dataset}/{chain}.json"
    resources:
        runtime=int(CONV_TIMEOUT)+60,
        threads=1,
        mem_mb=4000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {CONV_TIMEOUT}"\
        +" --n-steps {CONV_MAX_SAMPLES} --regression-deg {wildcards.d}"\
        +" --lambda-prop-std {wildcards.lstd}"


rule postprocess_sim_mcmc:
    input:
        pp=JULIA_PROJ_DIR+"/postprocess_samples.jl",
        raw=expand(RAW_DIR+"/mcmc_{{mcmc_settings}}/{{replicate}}/{chain}.json",
                   chain=SIM_CHAINS)
    output:
        out=PRED_DIR+"/mcmc_{mcmc_settings}/{replicate}.json"
    resources:
        runtime=3600,
        threads=1,
        mem_mb=6000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.pp} --chain-samples {input.raw}  --output-file {output.out}"

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
        +" --regression-deg {wildcards.d} --n-steps {SIM_MAX_SAMPLES}"\
        +" --lambda-prop-std 3.0 --large-indeg 15.0"


###############################
# Uniform MCMC
#
rule postprocess_sim_uniform_mcmc:
    input:
        pp=JULIA_PROJ_DIR+"/postprocess_samples.jl",
        raw=expand(RAW_DIR+"/uniform/{{replicate}}/{chain}.json",
                   chain=SIM_CHAINS)
    output:
        out=PRED_DIR+"/uniform/{replicate}.json"
    resources:
        runtime=3600,
        threads=1,
        mem_mb=6000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.pp} --chain-samples {input.raw}  --output-file {output.out}"


rule run_sim_uniform_mcmc:
    input:
        method=JULIA_PROJ_DIR+"/Catsupp.jl",
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv",
    output:
        RAW_DIR+"/uniform/{replicate}/{chain}.json"
    resources:
        runtime=SIM_TIMEOUT,
        threads=1,
        mem_mb=2000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {SIM_TIMEOUT}"\
        +" --regression-deg 1 --n-steps {SIM_MAX_SAMPLES}"\
        +" --lambda-prop-std 3.0 --large-indeg 15.0 --proposal uniform"


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

rule run_sim_lasso:
    input:
        method=JULIA_PROJ_DIR+"/lasso.jl",
	ts=TS_DIR+"/{replicate}.csv",
        ref=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/lasso/{replicate}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=100
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts} {input.ref} {output}"

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

rule dream_convergence_viz:
    input:
        expand(DREAM_PRED_DIR+"/mcmc_d={{d}}_lstd={{lstd}}/{{dataset}}_replicate={rep}.json",
               rep=CONV_REPLICATES) 
    output:
        FIG_DIR+"/dream/convergence/mcmc_d={d}_lstd={lstd}/{dataset}.png"
    script:
        SCRIPT_DIR+"/convergence_viz.py"

rule score_dream_mcmc:
    input:
        scorer=SCRIPT_DIR+"/score_dream.py",
        tr_desc=DREAM_TRU_DIR+"/TrueVec_{cell_line}_{stim}.csv", 
        preds=DREAM_PRED_DIR+"/{method}/cl={cell_line}_stim={stim}_replicate={rep}.json",
        ab=DREAM_REF_DIR+"/cl={cell_line}_antibodies.json",
    output:
        out=DREAM_SCORE_DIR+"/{method}/cl={cell_line}_stim={stim}_replicate={rep}.json" 
    resources:
        mem_mb=100,
        threads=1
    shell:
        "python {input.scorer} {input.preds} {input.tr_desc} {input.ab} {output.out}"


rule postprocess_dream_mcmc:
    input:
        pp=JULIA_PROJ_DIR+"/postprocess_samples.jl",
        raw=expand(DREAM_OUT_DIR+"/mcmc_{{mcmc_settings}}/{{replicate}}/chain={chain}.json",
                   chain=DREAM_CHAINS)
    output:
        out=DREAM_PRED_DIR+"/mcmc_{mcmc_settings}/{replicate}.json"
    resources:
        runtime=3600,
        threads=1,
        mem_mb=6000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.pp} --chain-samples {input.raw} --output-file {output.out}"\
        +" --stop-points {DREAM_STOPPOINTS}"


rule run_dream_mcmc:
    input:
        method=JULIA_PROJ_DIR+"/Catsupp.jl",
        ts_file=DREAM_PREP_TS_DIR+"/cl={cell_line}_stim={stimulus}.csv",
        ref_dg=DREAM_REF_DIR+"/cl={cell_line}.csv",
    output:
        DREAM_OUT_DIR+"/mcmc_d={d}_lstd={lstd}/cl={cell_line}_stim={stimulus}_replicate={replicate}/chain={chain}.json"
    resources:
        runtime=7200,
        threads=1,
        mem_mb=3000
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts_file} {input.ref_dg} {output} {DREAM_TIMEOUT}"\
        +" --n-steps {CONV_MAX_SAMPLES} --regression-deg {wildcards.d}"\
        +" --lambda-prop-std {wildcards.lstd} --large-indeg {MCMC_INDEG}"


rule score_dream_predictions:
    input:
        scorer=SCRIPT_DIR+"/score_dream.py",
        tr_desc=DREAM_TRU_DIR+"/TrueVec_{cell_line}_{stim}.csv", 
        preds=DREAM_PRED_DIR+"/{method}/cl={cell_line}_stim={stim}.json",
        ab=DREAM_REF_DIR+"/cl={cell_line}_antibodies.json",
    output:
        out=DREAM_SCORE_DIR+"/{method}/cl={cell_line}_stim={stim}.json" 
    resources:
        mem_mb=100,
        threads=1
    shell:
        "python {input.scorer} {input.preds} {input.tr_desc} {input.ab} {output.out}"


rule run_dream_funchisq:
    input: 
        method=FUNCH_DIR+"/funchisq_wrapper.R",
        ts_file=DREAM_PREP_TS_DIR+"/{replicate}.csv"
    output:
        DREAM_PRED_DIR+"/funchisq/{replicate}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=500
    shell:
        "Rscript {input.method} {input.ts_file} {output}"


rule run_dream_hill:
    input:
        ts_file=DREAM_PREP_TS_DIR+"/cl={cell_line}_stim={stim}.csv",
        ref_dg=DREAM_REF_DIR+"/cl={cell_line}.csv"
    output:
        DREAM_PRED_DIR+"/hill/cl={cell_line}_stim={stim}.json"
    resources:
        runtime=SIM_TIMEOUT+60,
        threads=1,
        mem_mb=2000
    shell:
        "matlab -nodesktop -nosplash -nojvm -singleCompThread -r \'cd(\"{HILL_DIR}\"); try, hill_dbn_wrapper(\"{input.ts_file}\", \"{input.ref_dg}\", \"{output}\", -1, \"auto\", {SIM_TIMEOUT}), catch e, quit(1), end, quit\'"


rule run_dream_lasso:
    input:
        method=JULIA_PROJ_DIR+"/lasso.jl",
        ref=DREAM_REF_DIR+"/cl={cell_line}.csv",
	ts=DREAM_PREP_TS_DIR+"/cl={cell_line}_stim={stim}.csv"
    output:
        DREAM_PRED_DIR+"/lasso/cl={cell_line}_stim={stim}.json"
    resources:
        runtime=60,
        threads=1,
        mem_mb=100
    shell:
        "julia --project={JULIA_PROJ_DIR} {input.method} {input.ts} {input.ref} {output}"


rule run_dream_prior_baseline:
    input:
        method=JULIA_PROJ_DIR+"/prior_baseline.jl",
        ref=DREAM_REF_DIR+"/cl={cell_line}.csv"
    output:
        DREAM_PRED_DIR+"/prior_baseline/cl={cell_line}_stim={stim}.json"
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
        expand(DREAM_PREP_TS_DIR+"/cl={{cell_line}}_stim={stimulus}.csv", stimulus=STIMULI)
    shell:
        "python scripts/preprocess_dream_ts.py {input} {DREAM_PREP_TS_DIR} --ignore-inhibitor"


rule preprocess_dream_prior:
    input:
        DREAM_PRIOR_DIR+"/{cell_line}.eda"
    output:
        edges=DREAM_REF_DIR+"/cl={cell_line}.csv",
        ab=DREAM_REF_DIR+"/cl={cell_line}_antibodies.json"
    shell:
        "python scripts/preprocess_dream_prior.py {input} {output.edges} {output.ab}"

# END EXPERIMENTAL DATA JOBS
##############################

