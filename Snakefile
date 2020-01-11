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
# Get the path of the Julia PackageCompiler
JULIAC_PATH = glob.glob(os.path.join(os.environ["HOME"],
                                     ".julia/packages/PackageCompiler/*/juliac.jl")
                        )[0]
SIM_REPLICATES = list(range(config["simulation_study"]["N"]))
SIM_GRID = config["simulation_study"]["simulation_grid"]
M = SIM_GRID["M"]
MC_PARAMS = config["mcmc_hyperparams"]
REG_DEGS = MC_PARAMS["regression_deg"]
CHAINS = list(range(MC_PARAMS["n_chains"]))
N_SAMPLES = MC_PARAMS["n_samples"]
THINNINGS = MC_PARAMS["thinning"]
BURNINS = MC_PARAMS["burnin"]
TIMEOUT = config["timeout"]

#############################
# HELPER FUNCTIONS
#############################


def get_mcmc_sim_scoring_inputs(wildcards, dirpath, ext):
    return ["{}_d={}_n={}_b={}_th={}/v={}_r={}_a={}_t={}_replicate={}.{}".format(dirpath,
	                                     wildcards.d,
					     wildcards.n,
					     wildcards.b,
					     wildcards.th,
                                             wildcards.v,
                                             wildcards.r,
                                             wildcards.a,
                                             wildcards.t,
                                             rep,ext) for rep in SIM_REPLICATES]

def get_scoring_inputs(wildcards, dirpath, ext):
    return ["{}/v={}_r={}_a={}_t={}_replicate={}.{}".format(dirpath,
                                          wildcards.v,
                                          wildcards.r,
                                          wildcards.a,
                                          wildcards.t,
                                          n,ext) for n in SIM_REPLICATES]

get_scoring_adjs = lambda wildcards, dirpath: get_scoring_inputs(wildcards, dirpath, "csv")
get_scoring_preds = lambda wildcards, dirpath: get_scoring_inputs(wildcards, dirpath, "json")

#############################
# RULES
#############################
rule simulation_study:
    input:
        expand("simulation-study/scores/mcmc_d={d}_n={n}_b={b}_th={th}/v={v}_r={r}_a={a}_t={t}.json", 
	        d=REG_DEGS, n=N_SAMPLES, b=BURNINS, th=THINNINGS, 
		v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]), 
	#expand("simulation-study/scores/funchisq/v={v}_r={r}_a={a}_t={t}.json",
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
        #expand("simulation-study/scores/hill/v={v}_r={r}_a={a}_t={t}.json",   
        #        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]),
	#        expand("simulation-study/scores/lasso/v={v}_r={r}_a={a}_t={t}.json",
	#        v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"])

rule simulate_data:
    input:
        simulator="builddir/simulate_data"
    output:
        "simulation-study/timeseries/v={v}_r={r}_a={a}_t={t}_replicate={rep}.tsv",
        "simulation-study/ref_dags/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
        "simulation-study/true_dags/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv"
    shell:
        "{input.simulator} {wildcards.v} {wildcards.r} {wildcards.a} {wildcards.t}"

######################
# MCMC JOBS
rule score_sim_mcmc:
    input:
        "builddir/scoring",
        tr_dg_fs=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/true_dags", "csv"), 
        pp_res=lambda wildcards: get_mcmc_sim_scoring_inputs(wildcards, "simulation-study/pred/mcmc", "json"), 
    output:
        out="simulation-study/scores/mcmc_d={d}_n={n}_b={b}_th={th}/v={v}_r={r}_a={a}_t={t}.json" 
    shell:
        "builddir/scoring --truth-files {input.tr_dg_fs} --pred-files {input.pp_res} --output-file {output.out}"

rule postprocess_sim_mcmc:
    input:
        "builddir/postprocess",
        raw=["simulation-study/raw/mcmc_d={d}_n={n}_b={b}_th={th}/v={v}_r={r}_a={a}_t={t}_replicate={rep}_chain="+str(c)+".json" for c in CHAINS]
    output:
        out="simulation-study/pred/mcmc_d={d}_n={n}_b={b}_th={th}/v={v}_r={r}_a={a}_t={t}_replicate={rep}.json"
    shell:
        "builddir/postprocess {input.raw} --output-file {output.out}"

rule perform_sim_mcmc:
    input:
        method="builddir/Catsupp",
        ts_file="simulation-study/timeseries/v={v}_r={r}_a={a}_t={t}_replicate={rep}.tsv",
        ref_dg="simulation-study/ref_dags/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv",
    output:
        out=list(["simulation-study/raw/mcmc_d={d}_n="\
			+str(n)+"_b="+str(b)+"_th="+str(th)\
			+"/v={v}_r={r}_a={a}_t={t}_replicate={rep}_chain={c}.json"\
			for n in N_SAMPLES for b in BURNINS for th in THINNINGS])
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} simulation-study/raw/ --n_samples "+" ".join([str(i) for i in N_SAMPLES])+" --burnin "+" ".join([str(b) for b in BURNINS])+" --thinning "+" ".join([str(th) for th in THINNINGS])+" --regression-deg {wildcards.d}"
        

# END MCMC JOBS
#####################

#####################
# FUNCHISQ JOBS
rule score_sim_funchisq:
    input: 
        "builddir/scoring",
        tr_dg_fs=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/true_dags", "csv"), 
        pp_res=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/pred/funchisq", "json"), 
    output:
        "simulation-study/scores/funchisq/v={v}_r={r}_a={a}_t={t}.json"
    shell:
        "builddir/scoring --truths {input.tr_dg_fs} --preds {input.pp_res} --out {output}"

rule perform_sim_funchisq:
    input: 
        "funchisq/funchisq_wrapper.R",
        ts_file="simulation-study/timeseries/v={v}_r={r}_a={a}_t={t}_replicate={rep}.tsv"
    output:
        "simulation-study/pred/funchisq/v={v}_r={r}_a={a}_t={t}_replicate={rep}.json"
    shell:
        "Rscript funchisq/funchisq_wrapper.R {input.ts_file} {output}"

# END FUNCHISQ JOBS
#####################

#####################
# HILL JOBS
rule score_sim_hill:
    input:
        "builddir/scoring",
        tr_dg_fs=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/true_dags", "csv"), 
        pp_res=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/pred/hill", "json"), 
    output:
        "simulation-study/scores/hill/v={v}_r={r}_a={a}_t={t}.json"
    shell:
        "builddir/scoring --truths {input.tr_dg_fs} --preds {input.pp_res} --out {output}"

rule perform_hill:
    input:
        "hill-method/hill_dbn_wrapper.m",
        ts_file="simulation-study/timeseries/v={v}_r={r}_a={a}_t={t}_replicate={rep}.tsv",
        ref_dg="simulation-study/ref_dags/v={v}_r={r}_a={a}_t={t}_replicate={rep}.csv"
    output:
        "simulation-study/pred/hill/v={v}_r={r}_a={a}_t={t}_replicate={rep}.json"
    shell:
        "matlab hill-method/hill_dbn_wrapper.m {input.ts_file} {input.ref_dg} {output}"\
        +" "+str(HILL_MAX_INDEG)+" "str(HILL_REG_MODE)+" "+str(TIMEOUT)

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

rule perform_lasso:
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
rule compile_simulator:
    input:
        "simulation-study/simulate_data.jl"
    output:
        "builddir/simulate_data"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae simulation-study/simulate_data.jl"

rule compile_postprocessor:
    input:
        "julia-project/postprocess.jl"
    output:
        "builddir/postprocess"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/postprocess.jl"


rule compile_scoring:
    input:
        "julia-project/scoring.jl"
    output:
        "builddir/scoring"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/scoring.jl"

rule compile_mcmc:
    input:
        "julia-project/Catsupp.jl"
    output:
        "builddir/Catsupp"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/Catsupp.jl"

rule compile_lasso:
    input:
        "julia-project/lasso.jl"
    output:
        "builddir/lasso"
    shell:
        "julia --project=julia-project/ {JULIAC_PATH} -vae julia-project/lasso.jl"

# END JULIA CODE COMPILATION
#############################

        
