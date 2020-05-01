#This script calculates the posterior probabilities for the training series using
#the h'(t) classifiers


#Define parameters
numbd <- 4

#Load sources
source("sources.R")
source("cvStratified.R")

for(earlynessperc in c(1:20)*5){

for(fold in c(1:5)){

#Load data
load(paste("../../Databases/UCR-",numbd,".RData",sep=""))
train<-database[[1]][database[[1]]$tt==0,]
train$tt<-NULL
classes<-as.factor(database[[2]][which(database[[1]]$tt==0)])

set.seed(123)
cv<-generateCVRuns(classes, ntimes=1, nfold=5, stratified=TRUE)
testindices<-cv[[1]][[fold]]
trainindices<-c(1:dim(train)[1])[-testindices]
traingp<-train[trainindices,]
testgp<-train[testindices,]


distance<-1
param<-0
kernel<-1

thetaestimate<-FALSE

#Calculate earlyness
earlyness<-round(earlynessperc*dim(traingp)[2]/100)

#Train SVM model
predicted<-trainmodel(traingp,classes[trainindices],testgp,classes[testindices],kernel=1, earlyness, distance, param, thetaestimate)
predicted<-as.data.frame(predicted)
predicted$class<-classes[testindices]


#Train GP model 
#DMtrain<-distanceMatrix(trainsvm, test=NULL, earlyness, distance, param)
#DMtest<-distanceMatrix(trainsvm, testsvm, earlyness, distance, param)
#model<-GP(DMtrain,classes[trainindices],DMtest,classes[testindices],kernel=1,estimatehyp=TRUE)
#predicted2<-model$Ptest


#Save values
fichero<-paste("../../Results/Probabilities/Train/Raw/probs-",numbd,"-",earlynessperc,"-",1,"-",fold,".txt",sep="")
write.table(predicted,fichero)

}
}

source("./formatting.probabilitiestrain.R")
