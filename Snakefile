configfile: "analysis_config.yaml"


rule all:
    input:
        "figures/"
        "outputs/"


rule make_figures:


rule convergence_analysis:
    output:
        "outputs/convergence.tsv"

rule sensitivity_analysis:
    output:
        "outputs/sensitivity.tsv"


rule simulation_study:
    output:
        "outputs/simulation.tsv"

rule real_data_analysis:
    output:
        
