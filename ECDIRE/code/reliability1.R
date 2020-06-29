#This function calculates the timestamps from the
#1st level reliability

reliability1<-function(accuracies, acc, numclus){
    
    accuracy<-matrix(nrow=numclus,ncol=20)
    
    #For each early timestamp
    for(earlyness in c(1:20)*5){
        #Collect the accuracy results
        aux = as.data.frame(accuracies[[earlyness]])
        #Calculate the mean accuracy of the cross validation
        accuracy[,earlyness/5]<-colMeans(aux,na.rm=TRUE)
        #For a more conservative option, calculate the minimum
        #accuracy[,earlyness/5]<-apply(aux,2,min,na.rm=TRUE)
        #Some corrections for classes that do not appear in the testing set
        accuracy[which(accuracy=="NaN")]<-0
    }
    
    timestamp<-c(1:dim(accuracy)[1])*0
    
    #Find when the accuracy trespasses the set threshold and calculate the timestamps
    for(i in c(1:dim(accuracy)[1])){
        if(acc[i]==0) {
            timestamp[i]<-NA
        } else {
            flag<-FALSE
            for(j in c(1:dim(accuracy)[2])){
                if(accuracy[i,j]>=acc[i] & flag==FALSE){
                    flag<-TRUE
                    timestamp[i]<-j
                }
                if(accuracy[i,j]<acc[i]){flag==FALSE}
            }
        }
    }
    timestamp[which(timestamp==0)]<-20
    timestamp<-timestamp*5
    
    return(timestamp)
}