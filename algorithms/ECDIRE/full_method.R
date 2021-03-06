### command line arguments:

## -data /path/to/train /path/tot/test
# data files should be in UCR data format: tsv with as first column class
# lables and in the remaining columns the time series

## -acc_perc the desired percentage of accuracy on the full lenght ts.
# controlls the tradeoff between earliness and accuracy

## -distance
# the distance function used.

## -np
# number of threads used in parallel crossvalidation step

## -seed
# seed used to initialize the random number generator

## -kernel
# the GP kernel that is used for the interal GP classifiers chose one of the following:
# 'gauss'     ##  Gaussian
# 'iprod'     ##  Inner product
# 'cauchy'    ##  Cauchy (heavy tailed) in distance
# 'laplace'   ##  Laplacian

# Get commandline arguments
suppressMessages(require('R.utils'));
suppressMessages(require('utils'));
defaults = list(doHPO = T,       # toggle GP HP search
                acc_perc = 100,
                distance = 'euclidean',
                dSigma = .2,
                dN = 20,
                dTau = 0,
                np = 4,
                seed = 0,
                kernel = 'iprod')
params = commandArgs(asValues=TRUE, adhoc=TRUE, defaults=defaults)
datapaths = base::commandArgs(trailingOnly=TRUE)[c(2,3)]

if (is.na(datapaths[2])) {
    stop(paste('Please provide train and test data as follows:\n    Rscript', params$file,
               '-data /path/to/train /path/to/test [params]'))
}

trainpath = datapaths[1]
testpath = datapaths[2]

setwd(dirname(params$file))
options(warn=1, showWarnCalls=T, showErrorCalls=T)

#Source all necessary internal files
suppressMessages(source("sources.R"))

# set number of cores for explicit parallelization
assign("num_cores", params$np, envir = .GlobalEnv)
# globally defined seed
assign("seed", params$seed, envir = .GlobalEnv)
set.seed(seed)

# suppress parallel BLAS routines
blas_set_num_threads(1)

# set distance metric definition
distance = list(metric=params$distance, sigma=params$dSigma,
				n=params$dN, tau=params$dTau)

start = proc.time()
#Train relGP classifier framework
out = relGP(trainpath, testpath, distance, params$kernel, params$doHPO, params$acc_perc)
earliness = if(!is.nan(out$meanearlyness)) out$meanearlyness/100 else 1
err = if(!is.nan(out$accuracy)) 1-out$accuracy else 1
cat(sprintf('Result: SUCCESS, %g, [%g, %g], 0\n', (proc.time() - start)['elapsed'], earliness, err))
