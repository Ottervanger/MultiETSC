### command line arguments:

## -data /path/to/train /path/tot/test
# data files should be in UCR data format: tsv with as first column class
# lables and in the remaining columns the time series

## -distance
# the distance function used
# 'euclidean'
# 'dtw'         ## dynamic time warping
# 'edr'         ## edit distance on real sequence
# 'fourier'

## -kernel
# the GP kernel that is used for the interal GP classifiers chose one of the following:
# 'gauss'     ##  Gaussian
# 'iprod'     ##  Inner product
# 'cauchy'    ##  Cauchy (heavy tailed) in distance
# 'laplace'   ##  Laplacian

## -optimizer
# the optimizer used to optimize the internal SR prameters
# 'ga'      ## genetic algorithm
# 'optim'   ## Nelder-Mead optimization
# 'sa'      ## Simulated Annealing
# 'pso'     ## particle swarm optimization

## -alpha
# parameter controlling tradeoff between earliness and accuracy
# real on the closed intervall [0,1]

## -sr
# stopping rule type
# {'sr1', 'sr2'}

## -reg
# regularisation type used in SR parameter optimization
# {'none', 'L0', 'L1'}

## -lambda
# quantifying amount of regularisation
# non-negative real

## -seed
# seed used to initialize the random number generator

## -np
# number of threads used in crossvalidation and optimizatino steps

##  Parameters to be exposed:
##  Parameters related to distance calculation. The distanceMatrix function takes
# in a 'param' which is used in multiple different ways and it at the moment simply
# fixed at 5. This should be exposed. Note this parameter might be expanded into
# different parameters, one for each role that 'param' takes. Also note that this
# parameter heavily influences computational complexity.

##  Parameters directly realated to computational effort committed. These include 
# the numer of optimization iterations, optimization of the GP regression hyperparams,
# and the number of iterations. Number of train tests splits in crossvalidation step.

# Get commandline arguments
suppressMessages(require('R.utils'));
defaults = list(
    distance = 'euclidean',  # distance metric {'euclidean', 'dtw', 'edr', 'fourier'}
    kernel = 'iprod',        # GP kernel {'iprod', 'gauss', 'cauchy', 'laplace'}

    optimizer = 'ga',        # optimization method
    alpha = 0.8,             # earliness accuracy tradeoff [0,1]
    sr = 'sr1',              # stopping rule
    # according to the paper regularization should not be applied when sr2 is used
    reg = 'none',            # 'none' for CF1, L0 for CF2, L1 for CF3
    lambda = 0,              # non-negative real amount of regularization, only used when reg != 'none'
    seed = 0,
    np = 12)

params = commandArgs(asValues=TRUE, adhoc=TRUE, defaults=defaults)
datapaths = base::commandArgs(trailingOnly=TRUE)[c(2,3)]

if (is.na(datapaths[2])) {
    stop(paste('Please provide train and test data as follows:\n    Rscript', params$file,
               '-data /path/to/train /path/to/test [params]'))
}

trainpath = datapaths[1]
testpath = datapaths[2]
options(warn=1, showWarnCalls=T, showErrorCalls=T)

setwd(dirname(dirname(params$file)))

#Source all necessary internal files
suppressMessages(source("prediction.R"))
suppressMessages(require("RhpcBLASctl"))

# suppress parallel BLAS routines
blas_set_num_threads(1)

start = proc.time()
# run
sink(file('/dev/null', open = 'wt'), type='message')
result = prediction(trainpath, testpath, distance=params$distance, kernel=params$kernel,
    optimizer=params$optimizer, alpha=params$alpha, sr=params$sr, reg=params$reg,
    lambda=params$lambda, np=params$np, seed=params$seed)
sink()

cat(sprintf('Result: SUCCESS, %g, [%g, %g], 0\n', (proc.time() - start)['elapsed'], result$earliness, 1-result$accuracy))
