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
RES_DIR = config["results_dir"]
COMP_DIR = "compiled"
FIG_DIR = "figures"

SIM_DIR = os.path.join(RES_DIR, "simulation_study")
SIMDAT_DIR = os.path.join(SIM_DIR, "datasets")
TS_DIR = os.path.join(SIMDAT_DIR, "timeseries")
REF_DIR = os.path.join(SIMDAT_DIR, "ref_graphs")
TRU_DIR = os.path.join(SIMDAT_DIR, "true_graph")
RAW_DIR = os.path.join(SIMDAT_DIR, "raw")
PRED_DIR = os.path.join(SIM_DIR, "predictions")
SCORE_DIR = os.path.join(SIM_DIR, "scores")

# Simulation study parameters
SIM_PARAMS = config["simulation_study"]
SIM_TIMEOUT = SIM_PARAMS["timeout"]
SIM_REPLICATES = list(range(SIM_PARAMS["N"]))
SIM_GRID = SIM_PARAMS["simulation_grid"]
SIM_M = SIM_GRID["M"]

# MCMC hyperparameters (for simulation study)
MC_PARAMS = SIM_PARAMS["mcmc_hyperparams"]
REG_DEGS = MC_PARAMS["regression_deg"]
CHAINS = list(range(MC_PARAMS["n_chains"]))
N_SAMPLES = MC_PARAMS["n_samples"]
THINNINGS = MC_PARAMS["thinning"]
BURNINS = MC_PARAMS["burnin"]

# Hill hyperparameters
HILL_PARAMS = SIM_PARAMS["hill_hyperparams"]
HILL_TIME_PARAMS = config["hill_timetest"]
HILL_TIME_COMBS = HILL_TIME_PARAMS["deg_v_combs"]
HILL_MODES = HILL_TIME_PARAMS["modes"]
HILL_TIME_TIMEOUT = HILL_TIME_PARAMS["timeout"]

#############################
# RULES
#############################
rule all:
    input:
        ## simulation study results
	#expand(SCORE_DIR+"/mcmc_d={d}_n={n}_b={b}_th={th}/v={v}_r={r}_a={a}_t={t}.json", 
	#       d=REG_DEGS, n=N_SAMPLES, b=BURNINS, th=THINNINGS, 
	#       v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]), 
	#expand(SCORE_DIR+"/funchisq/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand(SCORE_DIR+"/hill/v={v}_r={r}_a={a}_t={t}.json",   
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
	#expand("simulation-study/scores/lasso/v={v}_r={r}_a={a}_t={t}.json",
	#        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
	## Hill timetest results
        FIG_DIR+"/hill_method_timetest.png"	

rule simulate_data:
    input:
        simulator="builddir/simulate_data"
    output:
        ts=TS_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        ref=REF_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        true=TRU_DIR+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv"
    shell:
        "{input.simulator} {wildcards.v} {wildcards.t} {SIM_M} {wildcards.r} {wildcards.a} 3 {output.ref} {output.true} {output.ts}"

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

rule postprocess_sim_mcmc:
    input:
        "builddir/postprocess",
        raw=[RAW_DIR+"/mcmc_{mcmc_settings}/{replicate}_chain="+str(c)+".json" for c in CHAINS]
    output:
        out=PRED_DIR+"/mcmc_{mcmc_settings}/{replicate}.json"
    shell:
        "builddir/postprocess {input.raw} --output-file {output.out}"

rule run_sim_mcmc:
    input:
        method="builddir/Catsupp",
        ts_file=TS_DIR+"/{replicate}.csv",
        ref_dg=REF_DIR+"/{replicate}.csv",
    output:
        out=list([RAW_DIR+"/mcmc_d={d}_n="+str(n)+"_b="+str(b)+"_th="+str(th)\
			+"/{replicate}_chain={c}.json"\
			for n in N_SAMPLES for b in BURNINS for th in THINNINGS])
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} "+RAW_DIR\ 
        +" {wildcards.replicate}_chain={wildcards.c}.json"\
        +" --n-samples " + " ".join([str(i) for i in N_SAMPLES])\
        +" --burnin "+ " ".join([str(b) for b in BURNINS])\
        +" --thinning "+" ".join([str(th) for th in THINNINGS])\
        +" --regression-deg {wildcards.d}"\
        +" --timeout "+str(SIM_TIMEOUT)
        

# END MCMC JOBS
#####################

#####################
# FUNCHISQ JOBS

rule run_sim_funchisq:
    input: 
        "funchisq/funchisq_wrapper.R",
        ts_file=TS_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/funchisq/{replicate}.json"
    shell:
        "Rscript funchisq/funchisq_wrapper.R {input.ts_file} {output}"

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
        "matlab -nodesktop -nosplash -nojvm -r \'cd(\"hill-method\"); try, hill_dbn_wrapper(\"{input.ts_file}\", \"{input.ref_dg}\", \"{output}\", -1, \"full\", "+str(SIM_TIMEOUT)+"), catch e, quit(1), end, quit\'"


rule run_timetest_hill:
    input:
        ts=TS_DIR+"/{replicate}.csv",
	ref=REF_DIR+"/{replicate}.csv"
    output:
        PRED_DIR+"/hill_{deg}_{mode}/{replicate}.json"
    shell:
        "matlab -nodesktop -nosplash -nojvm -r \'cd(\"hill-method\"); try, hill_dbn_wrapper(\"{input.ts}\", \"{input.ref}\", \"{output}\", {wildcards.deg}, \"{wildcards.mode}\", "+str(HILL_TIME_TIMEOUT)+"), catch e, quit(1), end, quit\'"


rule figure_timetest_hill:
    input:
        [PRED_DIR+"/hill_"+str(comb[0])+"_"+str(m)+"/v="+str(comb[1])+"_r="+str(SIM_GRID["R"][0])+"_a="+str(SIM_GRID["A"][0])+"_t="+str(SIM_GRID["T"][0])+"_replicate=0.json" for comb in HILL_TIME_COMBS for m in HILL_MODES]
    output:
        FIG_DIR+"/hill_method_timetest.png"

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
        "builddir/simulate_data"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae simulation-study/simulate_data.jl"

rule compile_postprocessor:
    input:
        "builddir/Catsupp",
        "julia-project/postprocess.jl"
    output:
        "builddir/postprocess"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/postprocess.jl"


rule compile_scoring:
    input:
        "julia-project/scoring.jl",
        "builddir/postprocess"
    output:
        "builddir/scoring"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/scoring.jl"

rule compile_mcmc:
    input:
        "builddir/simulate_data",
        "julia-project/Catsupp.jl"
    output:
        "builddir/Catsupp"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/Catsupp.jl"

rule compile_lasso:
    input:
        "julia-project/lasso.jl",
        "builddir/simulate_data"
    output:
        "builddir/lasso"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/lasso.jl"

# END JULIA CODE COMPILATION
#############################

        
