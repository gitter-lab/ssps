# Sparse Signaling Pathway Sampling
[![Test SSPS](https://github.com/gitter-lab/ssps/actions/workflows/test.yml/badge.svg)](https://github.com/gitter-lab/ssps/actions/workflows/test.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3939287.svg)](https://doi.org/10.5281/zenodo.3939287)

Code related to the manuscript _[Inferring signaling pathways with probabilistic programming](https://doi.org/10.1093/bioinformatics/btaa861)_ (Merrell & Gitter, 2020) Bioinformatics, 36:Supplement_2, i822–i830.

This repository contains the following:

* `SSPS`: A method that infers relationships between variables using time series data.
    - Modeling assumption: the time series data is generated by a Dynamic Bayesian Network (DBN).
    - Inference strategy: MCMC sampling over possible DBN structures.
    - Implementation: written in Julia, using the [`Gen` probabilistic programming language](https://probcomp.github.io/Gen/)
* Analysis code:
    - simulation studies;
    - convergence analyses;
    - evaluation on experimental data;
    - a [Snakefile](https://snakemake.readthedocs.io/en/stable/#) for managing all of the analyses.
    
# Installation and basic setup

(If you plan to reproduce *all* of the analyses, then make sure you're on a host with access to plenty of CPUs.
Ideally, you would have access to a cluster of some sort.)

1. **Clone** this repository
```
git clone git@github.com:gitter-lab/ssps.git
```
2. Install **Julia 1.4** (and all Julia dependencies)
    * Download the correct Julia binary here: https://julialang.org/downloads/. <br>
      E.g., for Linux x86_64:
    ```
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.2-linux-x86_64.tar.gz
    $ tar -xvzf julia-1.4.2-linux-x86_64.tar.gz
    ```
    * Find additional installation instructions here: https://julialang.org/downloads/platform/.
    * Use `Pkg` -- Julia's package manager -- to install the project's julia dependencies:
    ```
    $ cd ssps/SSPS
    $ julia --project=. 
                   _
       _       _ _(_)_     |  Documentation: https://docs.julialang.org
      (_)     | (_) (_)    |
       _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 1.4.2 (2020-05-23)
     _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
    |__/                   |

    julia> using Pkg
    julia> Pkg.instantiate()
    julia> exit()
    ```


# Reproducing the analyses

In order to reproduce the analyses, you will need some extra bits of software.
* We use [Snakemake](https://snakemake.readthedocs.io/en/stable/#) -- a python package -- to manage the analysis workflow.
* We use some other python packages to postprocess the results, produce plots, etc.
* Some of the baseline methods are implemented in R or MATLAB.


Hence, the analyses entail some extra setup:
    
1. Install **python3.7 dependencies (using `conda`)**
    * For the purposes of these instructions, we assume you have Anaconda3 or Miniconda3 installed,
      and have access to the `conda` environment manager. <br>
      (We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html);
      [find full installation instructions here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).) 
    * We recommend setting up a dedicated virtual environment for this project:
    ```
    $ conda env create -n my_environment
    $ conda activate my_environment
    (my_environment) $ 
    ```
    * Make sure Snakemake, pandas, numpy, and matplotlib are installed.
    ```
    (my_environment) $ conda install -c bioconda -c conda-forge snakemake
    (my_environment) $ conda install pandas matplotlib numpy
    ```
    * If you plan to reproduce the analyses **on a cluster**, then install 
    [cookiecutter](https://cookiecutter.readthedocs.io/en/1.7.0/)...
    ```
    conda install -c conda-forge cookiecutter
    ```
    and find the appropriate *Snakemake profile* from this list:
    https://github.com/Snakemake-Profiles/doc
    install the *Snakemake profile* using cookiecutter:
    ```
    (my_environment) $ cookiecutter https://github.com/OWNER/PROFILE.git
    ```
    
2. Install **R packages**
    * [Ckmeans.1d.dp](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html)
    * [FunChisq](https://cran.r-project.org/web/packages/FunChisq/index.html)

3. Check whether **MATLAB** is installed.
    * If you don't have MATLAB, then you won't be able to run the 
    [exact DBN inference method of Hill et al., 2012](https://academic.oup.com/bioinformatics/article/28/21/2804/235527).
    * You'll need to comment out the `hill` method wherever it appears in `analysis_config.yaml`.

After completing this additional setup, we are ready to **run the analyses**.
1. Make any necessary modifications to the configuration file: `analysis_config.yaml`.
   This file controls the space of hyperparameters and datasets explored in the analyses.
2. Run the analyses using `snakemake`:
    * If you're running the analyses on your local host, simply move to the directory containing `Snakefile`
    and call `snakemake`.
    ```
    (my_environment) $ cd ssps
    (my_environment) $ snakemake
    ```
    * Since Julia is a dynamically compiled language, some time will be devoted to compilation when you run SSPS for the first time. You may see some warnings in `stdout` -- this is normal.
    * If you're running the analyses on a cluster, call snakemake with the same **Snakemake profile** you found 
    [here](https://github.com/Snakemake-Profiles/doc):
    ```
    (my_environment) $ cd ssps
    (my_environment) $ snakemake --profile YOUR_PROFILE_NAME
    ```
    (You will probably need to edit the job submission parameters in the profile's `config.yaml` file.)
4. Relax. It will take tens of thousands of cpu-hours to run all of the analyses.


# Running SSPS on your data

Follow these steps to run SSPS on your dataset. You will need
* a CSV file (tab separated) containing your time series data
* a CSV file (comma separated) containing your prior edge confidences.
* Optional: a JSON file containing a list of variable names (i.e., node names).

1. Install the **python3.7 dependencies** if you haven't already. Find detailed instructions above.
2. `cd` to the `run_ssps` directory
3. Configure the parameters in `ssps_config.yaml` as appropriate
4. Run Snakemake: `$ snakemake --cores 1`. Increase 1 to increase the maximum number of CPU cores to be used.

## A note about parallelism 

SSPS allows two levels of parallelism: (1) at the Markov chain level and (2) at the iteration level.
* Chain-level parallelism is provided via Snakemake. For example, Snakemake can run 4 chains simultaneously if you specify `--cores 4` at the command line: `$ snakemake --cores 4`. In essence, this just creates 4 instances of SSPS that run simultaneously.
* Iteration-level parallelism is provided by [Julia's multi-threading features](https://docs.julialang.org/en/v1/manual/multi-threading/). The number of threads available to a SSPS instance is specified by an environment variable: `JULIA_NUM_THREADS`.
* The total number of CPUs used by your SSPS jobs is **the product** of Snakemake's `--cores` parameter and Julia's `JULIA_NUM_THREADS` environment variable. Concretely: if we run `snakemake --cores 2` and have `JULIA_NUM_THREADS=4`, then up to 8 CPUs may be used at one time by the SSPS jobs.

# Licenses

SSPS is available under the [MIT License](LICENSE.txt), Copyright © 2020 David Merrell.

The MATLAB code [`dynamic_network_inference.m`](hill-method/dynamic_network_inference.m) has been modified from the [original version](http://mukherjeelab.nki.nl/DBN.html), Copyright © 2012 Steven Hill and Sach Mukherjee.

The `dream-challenge` data is described in [Hill et al., 2016](http://doi.org/10.1038/nmeth.3773) and is originally from [Synapse](https://www.synapse.org/#!Synapse:syn1720047/wiki/93228).
