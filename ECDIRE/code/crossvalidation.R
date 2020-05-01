
crossvalidation<-function(databasename,distance,kernel,estimatehyp){
#LOAD DATA
data<-loadData(databasename)
trainclass<-data[[1]]$class
train<-data[[1]]
train$class<-NULL
rm(data)

#CREATE the sets for the 10 times repeat 5-fold cross validation.
set.seed(100)
cv<-generateCVRuns(as.numeric(trainclass),ntimes=10,nfold=5,stratified=TRUE)

#Initilize results
acc<-matrix(nrow=0,ncol=numclus)
probabilities<-matrix(nrow=0,ncol=numclus+1)

for(earlynessperc in c(1:20)*5){
for(rep in 1:10){
for(fold in 1:5){

#CREATE DATASETS for this repetition and fold
traincv<-train[-cv[[rep]][[fold]],]
testcv<-train[cv[[rep]][[fold]],]
trainclasscv<-trainclass[-cv[[rep]][[fold]]]
testclasscv<-trainclass[cv[[rep]][[fold]]]

#CALCULATE early timestamp
earlyness<-earlynessperc*(dim(train)[2])/100

#CALCULATE DISTANCE MATRIX for given early timestamp.
DMtrain<-distanceMatrix(train=traincv, earlyness=earlyness, distance=distance)
DMtest<-distanceMatrix(train=traincv, test=testcv, earlyness=earlyness, distance=distance)

#TRAIN the GP . Hyperparameter estimation set to true
model<-GP(DMtrain,trainclasscv,DMtest,testclasscv,kernel,estimatehyp)

#EXTRACT the accuracy for each class
accaux<-obtainaccuracy(model)
acc<-rbind(acc,accaux)

#EXTRACT  of the posterior probabilities of the correctly classified series
correct<-which(predClass(model,numclus)==testclasscv)
probabilitiesaux<-matrix(nrow=length(correct),ncol=numclus+1)
probabilitiesaux[,(1:numclus)]<-model$Ptest[correct,]
probabilitiesaux[,(numclus+1)]<-testclasscv[correct]
probabilities<-rbind(probabilities,probabilitiesaux)

}
}

#SAVE results

file1<-paste(getwd(),"/results/accuracies/acc-",databasename,"-",earlynessperc,".txt",sep="")
write.table(acc,file=file1,row.names=FALSE, col.names=FALSE)
file2<-paste(getwd(),"/results/probabilities/prob-",databasename,"-",earlynessperc,".txt",sep="")
write.table(probabilities,file=file2, row.names=FALSE, col.names=FALSE)
}

}

