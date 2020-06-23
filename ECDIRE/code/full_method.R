
# Get commandline arguments
suppressMessages(require('R.utils'));
suppressMessages(require('utils'));
defaults = list(validation = FALSE,
                HPO = TRUE,
                acc_perc = 100,
                kernel = 1)
params = commandArgs(asValues=TRUE, 
                     adhoc=TRUE,
                     defaults=defaults)
datapaths = base::commandArgs(trailingOnly=TRUE)[c(2,3)]

setwd(dirname(dirname(params$file)))

trainpath = datapaths[1]
testpath = datapaths[2]

#Cross validation process necessary?
cvprocess = params$validation

#Hyperparameter estimation necessary?
estimatehyp = params$HPO

#Choose derired level of accuracy
accuracythreshold = params$acc_perc

#Distance measure (Only Euclidean distance implemented)
distance<-1

#DEFINE KERNEL (INNER PRODUCT)
kernel = params$kernel

#Source all necessary internal files
source("code/sources.R")

# cross validation is used to get estimate of full len accuracy
if(cvprocess){
    crossvalidation(databasename,distance, kernel, estimatehyp)
}

#Train relGP classifier framework and save results in results/finalresults
relGP(trainpath, testpath, distance, kernel, estimatehyp)
