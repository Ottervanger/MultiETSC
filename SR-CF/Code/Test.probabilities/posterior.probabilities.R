#This script calculates the posterior probabilities of the second
#half of the training set given the SVM-s trained with the first half of the 
#series

#Parameters
numbd<-4
#Load sources
source("sources.R")

for(earlynessperc in c(1:20)*5){


#Load data
load(paste("../../Databases/UCR-",numbd,".RData",sep=""))
train<-database[[1]][database[[1]]$tt==0,]
train$tt<-NULL
classestrain<-as.factor(database[[2]][which(database[[1]]$tt==0)])

test<-database[[1]][database[[1]]$tt==1,]
test$tt<-NULL
classestest<-as.factor(database[[2]][which(database[[1]]$tt==1)])



distance<-1
param<-0
kernel<-1

thetaestimate<-FALSE
#Calculate earlyness
earlyness<-round(earlynessperc*dim(train)[2]/100)

#Train SVM model
predicted<-trainmodel(train,classestrain,test,classestest,kernel=1, earlyness, distance, param, thetaestimate)
predicted<-as.data.frame(predicted)
predicted$class<-classestest


#Save values
fichero<-paste("../../Results/Probabilities/Test/Raw/probs-",numbd,"-",earlynessperc,".txt",sep="")
write.table(predicted,fichero)

}

source("formatting.probabilitiestest.R")

