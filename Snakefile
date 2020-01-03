# Snakefile
# 2019-01
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
SIM_DIRS = config["simulation_study"]["dirs"]

#############################
# HELPER FUNCTIONS
#############################
def get_scoring_inputs(wildcards, dirpath, ext):
    return ["{}/{}_{}_{}_{}_{}.{}".format(dirpath,
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
        expand("simulation-study/scores/mcmc/{v}_{r}_{a}_{t}.json", 
                v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"]) 
        expand("simulation-study/scores/funchisq/{v}_{r}_{a}_{t}.json",
                v=SIM_GRID["V"], r=SIM_GRID["R"], a=SIM_GRID["A"], t=SIM_GRID["T"])
        #expand("simulation-study/scores/hill/{}_{}_{}_{}.json")   # TODO hill scores
        #expand("simulation-study/scores/lasso/{}_{}_{}_{}.json"), # TODO lasso scores

rule simulate_data:
    input:
        simulator="builddir/simulate_data"
    output:
        temp("simulation-study/timeseries/{v}_{r}_{a}_{t}_{n}.tsv"),
        temp("simulation-study/ref_dags/{v}_{r}_{a}_{t}_{n}.csv"),
        temp("simulation-study/true_dags/{v}_{r}_{a}_{t}_{n}.csv")
    shell:
        "{input.simulator} {wildcards.v} {wildcards.r} {wildcards.a} {wildcards.t}"

######################
# MCMC JOBS
rule score_sim_mcmc:
    input:
        "builddir/scoring",
        tr_dg_fs=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/true_dags", "csv"), 
        pp_res=lambda wildcards: get_scoring_inputs(wildcards, "simulation-study/mcmc_postprocess", "json"), 
    output:
        out="simulation-study/scores/mcmc/{v}_{r}_{a}_{t}.json" # TODO MCMC scores
    shell:
        "builddir/scoring --truths {input.tr_dg_fs} --preds {input.pp_res} --out {output.out}"


rule postprocess_sim_mcmc:
    input:
        "builddir/postprocess",
        raw="simulation-study/raw_mcmc/{v}_{r}_{a}_{t}_{n}.json"
    output:
        out=temp("simulation-study/mcmc_postprocess/{v}_{r}_{a}_{t}_{n}.json")
    shell:
        "builddir/postprocess {input.raw} {output.out}"

rule perform_sim_mcmc:
    input:
        method="builddir/Catsupp",
        ts_file="simulation-study/timeseries/{v}_{r}_{a}_{t}_{n}.tsv",
        ref_dg="simulation-study/ref_dags/{v}_{r}_{a}_{t}_{n}.csv",
    output:
        out=temp("simulation-study/raw_mcmc/{v}_{r}_{a}_{t}_{n}.json")
    shell:
        "{input.method} {input.ts_file} {input.ref_dg} {output.out}"

# END MCMC JOBS
#####################

#####################
# FUNCHISQ JOBS
rule score_sim_funchisq:
    input:
    output:
        "simulation-study/scores/funchisq/{v}_{r}_{a}_{t}.json"
    shell:

rule perform_funchisq:
    input:
    output:
    shell:

# END FUNCHISQ JOBS
#####################

#####################
# HILL JOBS
rule score_sim_hill:
    input:
    output:
        "simulation-study/scores/hill/{v}_{r}_{a}_{t}.json"
    shell:
        ""

rule perform_hill:
    input:
    output:
    shell:
        ""

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

rule perform_hill:
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

        
