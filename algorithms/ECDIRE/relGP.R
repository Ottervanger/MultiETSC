relGP<-function(trainpath, testpath, distance, kernel, estimatehyp, accuracythreshold){

#COLLECT RELIABILITIES
reliabilities = reliability(trainpath, distance, kernel, estimatehyp, accuracythreshold)
reliability1 = reliabilities[[1]]
reliability2 = as.data.frame(reliabilities[[2]])
# since the reliability function consumes random numbers,
# depening on wether data is cached, we need to reset the seed
set.seed(seed+1)

#LOAD DATA
data = loadData(testpath)
testclass = as.numeric(levels(data$class))[data$class]
test = data$ts

data = loadData(trainpath)
trainclass = as.numeric(levels(data$class))[data$class]
train = data$ts

if (length(trainclass) > 300)
    blas_set_num_threads(2)

#CREATE TIMELINE
timestamps<-unique(reliability1[order(reliability1)])
timestamps<-timestamps[!is.na(timestamps)]
listaclasses<-list()
for(i in c(1:length(timestamps))){
  listaclasses[[i]]<-which(reliability1==timestamps[i])
}
timestamps2<-timestamps
timestamps<-length(train[1,])*timestamps/100

#INITIALIZE OBJECTS
#indices of  the series considered in each timestamp.
indicestrain<-c(1:dim(train)[1])
indicestest<-c(1:dim(test)[1])
#classes that we remove in each step
removeclasses<-c()
#statistics about the number of series classified and 
#number of correctly classified series
numseries<-c(1:length(timestamps))*NA
numcorrectseries<-c(1:length(timestamps))*NA
#class results for the test data
results<-rep(NA,length(testclass))


#FOR EACH RELEVANT TIMESTAMP
targetclasses<-c()
  
for(i in c(1:length(listaclasses))){

    #DEFINE target classes in this timestamp
    targetclasses<-c(targetclasses,listaclasses[[i]])

    #If no instances are left in the testing set, the process is ended
    if(length(indicestest)==0)
        break;

    #TESTING SET DATA AND CLASSES
    testaux<-test[indicestest,]
    testauxclass<-testclass[indicestest]
      
    #SAVE CLASSES AS FACTORS
    testauxclass<-factor(testauxclass)
    trainauxclass<-factor(trainclass) 
      
    #CREATE THE DISTANCE MATRICES
    DMtest<-distanceMatrix(train=train, test=testaux, earlyness=timestamps[i], distance=distance)

    #CREATE THE CLASSIFIER
    model<-GP(distanceMatrices[[timestamps2[i]/5]],trainclass,DMtest,testauxclass,kernel,estimatehyp)

    #CLASS PREDICTIONS
    classes<-predClass(model,numclus)  
    classes<-factor(classes, levels=levels(trainauxclass))
      
    #CHECK SECOND LEVEL RELIABILITY
    root<-FALSE
    if(i==length(listaclasses)){root<-TRUE}
    aux<-checkrel2(model,targetclasses,classes,reliability2[timestamps2[i]/5,],root,numclus)
    classified<-aux[[1]]
    results[indicestest[classified]]<-aux[[2]]
      
      
    #STATISTICS
    numseries[i]<-length(which(!is.na(results)))/dim(test)[1]
    numcorrectseries[i]<-length(which(results==testclass))/dim(test)[1]
        
      
    #CALCULATE THE NEW TEST INDEXES OF ELEMENTS THAT HAVE NOT BEEN CLASSIFIED YET
    if (is.null(classified))
        next;
    indicestest<-indicestest[-classified]
  
}



# NOW ALL CLASSES ARE SAFE AND WE APPLY A MULTICLASS CLASSIFIER
#UNTIL WE HAVE THE WHOLE SERIES OR WE HAVE CLASSIFIED ALL

earlynessperc<-max(timestamps2)+5

# GO-TODO: I think this is where the number of classifiers is hard-coded. 
# This needs to be turned into a parameter.
while(earlynessperc<=100 && length(indicestest)!=0 ){
    earlyness<-earlynessperc*length(train[1,])/100
    #TESTING SET DATA AND CLASSES
    testaux<-test[indicestest,]
    testauxclass<-testclass[indicestest]
    
    #TRAINING SET
    trainclass<-factor(trainclass)
    
    #CREATE THE DISTANCE MATRICES
    DMtest<-distanceMatrix(train=train, test=testaux, earlyness=earlyness, distance=distance)
    
    #CREATE THE CLASSIFIER
    model<-GP(distanceMatrices[[earlynessperc/5]],trainclass,DMtest,testauxclass,kernel,estimatehyp)
    classes<-predClass(model,numclus)
    classes<-factor(classes, levels=levels(trainclass))

    #CHECK SECOND LEVEL RELIABILITY
    aux<-checkrel2(model,as.numeric(levels(classes)),classes,reliability2[earlynessperc/5,],root=TRUE,numclus)
    classified<-aux[[1]]
    results[indicestest[classified]]<-aux[[2]]
    
    #SET CLASSES 
    numseries<-c(numseries,length(which(!is.na(results)))/dim(test)[1])
    numcorrectseries<-c(numcorrectseries,length(which(results==testclass))/dim(test)[1])
    
    #CALCULATE THE NEW INDEXES!! 
    if(length(classified)!=0){
        indicestest<-indicestest[-classified]
    }
    
    timestamps2<-c(timestamps2,earlynessperc)
    earlynessperc<-earlynessperc+5
}

#CALCULATE STATISTICS
numcorrectseries<-numcorrectseries[!is.na(numcorrectseries)] #Number of correctly classified series in each step (%)
accuracy<-numcorrectseries[length(numcorrectseries)] #Accuracy of the whole process
numseries<-numseries[!is.na(numseries)] #Number of classified series in each step (%)
timestamps2<-timestamps2[1:length(numseries)]#Timestamps in which series are classified
meanearlyness<-sum(timestamps2*c(numseries[1],diff(numseries))/numseries[length(numseries)]) #Mean earlyness
resultOut<-list(
    accuracy=accuracy,
    meanearlyness=meanearlyness,
    numseries=numseries,
    timestamps2=timestamps2,
    numcorrectseries=numcorrectseries) #Array of results

return(resultOut)
}

