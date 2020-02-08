# Catsupp

TODO: probably choose a different name

This repository contains the following:

* `Catsupp`: A method that infers relationships between variables using time series data.
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
git clone git@github.com:dpmerrell/graph-ppl.git
```
2. Install **Julia 1.2.0** (and all Julia dependencies)
    * Download the correct Julia 1.2.0 binary here: https://julialang.org/downloads/oldreleases/. <br>
      E.g., for Linux x86_64:
    ```
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.2/julia-1.2.0-linux-x86_64.tar.gz
    $ tar -xvzf julia-1.2.0-linux-x86_64.tar.gz
    ```
    * Find additional installation instructions here: https://julialang.org/downloads/platform/.
    * Use `Pkg` -- Julia's package manager -- to install the project's julia dependencies:
    ```
    $ cd graph-ppl/julia-project
    $ julia --project=. 
                   _
       _       _ _(_)_     |  Documentation: https://docs.julialang.org
      (_)     | (_) (_)    |
       _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
      | | | | | | |/ _` |  |
      | | |_| | | | (_| |  |  Version 1.2.0 (2019-08-20)
     _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
    |__/                   |

    julia> using Pkg
    julia> Pkg.instantiate()
    julia> exit()
    ```

3. (optional) **Compile** a `Catsupp` executable
    * If you intend to run the analyses, skip this step.
    * On the other hand: if you would like to use `Catsupp` *without* running the analyses, 
      then you can build an executable binary using the 
      [`PackageCompiler` julia package](https://github.com/JuliaLang/PackageCompiler.jl):
      ```
      $ cd graph-ppl/julia-project
      $ julia --project=. ~/.julia/packages/PackageCompiler/CJQcs/juliac.jl -vae Catsupp.jl
      ```
      This will leave an executable binary (`Catsupp`) located in the `./builddir/` directory.
      
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
    * You'll

After completing this additional setup, we are ready to **run the analyses**.
1. Make any necessary modifications to the configuration file: `analysis_config.yaml`.
   This file controls the space of hyperparameters and datasets explored in the analyses.
2. Run the analyses using `snakemake`:
    * If you're running the analyses on your local host, simply move to the directory containing `Snakefile`
    and call `snakemake`.
    ```
    (my_environment) $ cd graph-ppl
    (my_environment) $ snakemake
    ```
    * If you're running the analyses on a cluster, call snakemake with the same **Snakemake profile** you found 
    [here](https://github.com/Snakemake-Profiles/doc):
    ```
    (my_environment) $ cd graph-ppl
    (my_environment) $ snakemake --profile
    ```
    (You will probably need to edit the job submission parameters in the profile's `config.yaml` file.)
3. Relax. It will probably take some time to run all of the analyses.