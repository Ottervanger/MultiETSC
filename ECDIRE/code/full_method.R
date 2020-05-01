
#Set path to folder
setwd("")

#Define database name
databasename<-"UCR-4"

#Cross validation process necessary?
cvprocess<-FALSE

#Hyperparameter estimation necessary?
estimatehyp<-TRUE

#Choose derired level of accuracy
accuracythreshold<-100

#Distance measure (Only Euclidean distance implemented)
distance<-1

#DEFINE KERNEL (INNER PRODUCT)
kernel<-1

#Source all necessary internal files
source("code/sources.R")

#EXECUTE relGP METHOD

  #If necessary, execute cross validation process
  if(cvprocess){
    crossvalidation(databasename,distance, kernel, estimatehyp)
  }

  #Extract first and second level reliability information
  reliability(databasename, accuracythreshold)

  #Train relGP classifier framework and save results in results/finalresults
  relGP(databasename, distance, kernel, estimatehyp)
