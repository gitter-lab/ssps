# funchisq_wrapper.R
# 2019-12-16
# David Merrell
#
# This script wraps the FunChisq causal inference technique
# of Song et al. It performs the following steps:
# 1) read and preprocess a timeseries dataset using
#    `mclust` and `Ckmeans.1d.dp`, as described in the 
#    supplementary materials of Hill et al. 2016 
#    ("community-based effort")
# 2) produce a matrix of fun-chisquare scores for the full set of
#    pairwise functional dependency tests
# 3) outputs all relevant information to a JSON file


require("FunChisq")
require("Ckmeans.1d.dp")
require("mclust")
require("jsonlite")

# Load the time series file 
load_time_series <- function(ts_filename){

    ts_table = read.csv(ts_filename, sep="\t")

    # It turns out we don't have to distinguish between
    # timecourses or timesteps -- the functional dependence
    # test doesn't assume e.g. continuity. So we just return
    # all the samples (excluding the timecourse/timestep cols)
    ts_table[,-c(1,2)]
}


# Discretize the time series data as described in the 
# supplementary material for Hill et al. 2016
discretize_time_series <- function(ts_df){

    ncols = dim(ts_df)[2]
    for (i in 1:ncols){

    # Use Mclust to estimate a number of quantization levels
        mcl_result = Mclust(ts_df[,i])
        quanta = ifelse(mcl_result$G == 1, 3, mcl_result$G)

        # Use Ckmeans.1d.dp to discretize each column
    ckm_result = Ckmeans.1d.dp(ts_df[,i], k=quanta)
    ts_df[,i] = ckm_result$cluster
    }

    ts_df
}


# Run FunChisq on the discretized dataset
run_funchisq <- function(discretized_df){

    nvars = dim(discretized_df)[2]
   
    # Collect the statistics/pvalues/estimates
    fcsq_statistics = array(0,dim=c(nvars,nvars))
    fcsq_p_values = array(0,dim=c(nvars,nvars))
    fcsq_estimates = array(0,dim=c(nvars,nvars))

    for (parent in 1:nvars){
        for (child in 1:nvars){
            contingency = table(discretized_df[,parent],
                                discretized_df[,child])

            fcsq_result = fun.chisq.test(contingency, method="nfchisq")

            fcsq_statistics[parent,child] = fcsq_result$statistic
            fcsq_p_values[parent,child] = fcsq_result$p.value
            fcsq_estimates[parent,child] = fcsq_result$estimate
    }
    }

    return(list("statistic"=fcsq_statistics, 
                "value"=fcsq_p_values, 
                "edge_conf_key"="value",
                "estimate"=fcsq_estimates))

}


# Dump results to a JSON file
dump_results <- function(output_filename, fcsq_result){

    fileConn = file(output_filename)
    writeLines(toJSON(fcsq_result), fileConn)
    close(fileConn)

}

# Unpack command line args
args = commandArgs(trailingOnly=TRUE)
timeseries_filename = args[1]
output_filename = args[2]

# load data
ts_df = load_time_series(timeseries_filename)

# discretize data
discretized_df = discretize_time_series(ts_df)

# run FunChisq on all pairs of variables
fcsq_result = run_funchisq(discretized_df)

# Dump the results to a JSON file
dump_results(output_filename, fcsq_result)

