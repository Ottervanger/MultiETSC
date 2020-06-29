
# Get commandline arguments
suppressMessages(require('R.utils'));
suppressMessages(require('utils'));
defaults = list(doHPO = TRUE,       # do hyperparameter estimation?
                acc_perc = 100,     # desired level of accuracy
                distance = 1,       # 1 = Euclidean (only one implemented)
                kernel = 1)         # 1 = INNER PRODUCT
params = commandArgs(asValues=TRUE, adhoc=TRUE, defaults=defaults)
datapaths = base::commandArgs(trailingOnly=TRUE)[c(2,3)]

if (is.na(datapaths[2])) {
    stop(paste('Please provide train and test data as follows:\n    Rscript', params$file,
               '-data /path/to/train /path/to/test [params]'))
}

trainpath = datapaths[1]
testpath = datapaths[2]

setwd(dirname(dirname(params$file)))

#Source all necessary internal files
source("code/sources.R")

errorCallback = function() {
    Sys.sleep(1)
    traceback()
    recover()
    debugger()
}
options(error=errorCallback, warn=2)

#Train relGP classifier framework
out = relGP(trainpath, testpath, params$distance, params$kernel, params$doHPO, params$acc_perc)
saveRDS(out, file='output.rds', compress=FALSE)
# GO-TODO:
# Print the results in configurator-friendly ouput.
