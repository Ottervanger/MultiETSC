reliability<-function(databasename,accuracythreshold){
  
#COLLECT FULL LENGTH ACCURACY 

file<-paste(getwd(),"/results/accuracies/acc-",databasename,"-100.txt",sep="")
accuracy<-read.table(file)
accuracy<-colMeans(accuracy,na.rm=TRUE)
accuracy[which(accuracy=="NaN")]<-0
#We remove the classes that have an accuracy lower than 1/|C|
accuracy[which(accuracy<1/length(accuracy))]<-0 
numclus<-length(accuracy)
  
#CALCULATE LEVEL OF ACCURACY (% of the total accuracy)
acc<-accuracy*accuracythreshold/100

#CALCULATE RELIABILITY 1 
rel1<-reliability1(databasename,numclus,acc)

file<-paste(getwd(),"/results/reliabilities/rel1-",databasename,".txt",sep="")
write.table(rel1,file=file)

#EXTRACT THRESHOLDS FOR RELIABILITY 2
rel2<-reliability2(databasename, numclus)
file<-paste(getwd(),"/results/reliabilities/rel2-",databasename,".txt",sep="")
write.table(rel2,file=file)

}
