# This function calculates the timestamps from the
# 1st level reliability
reliability1 = function(accuracies, acc){
    accuracy = unname(sapply(accuracies, function(i) i))
    nClassifiers = dim(accuracy)[2]
    nClasses = dim(accuracy)[1]
    timestamp = rep(nClassifiers,nClasses)
    
    # Find when the accuracy surpasses the set threshold
    for(i in 1:nClasses){
        if(acc[i]==0) {
            timestamp[i]<-NA
            next
        } 
        timestamp[i] = which(accuracy[i,] >= acc[i])[1]
    }
    timestamp = timestamp*100/nClassifiers
    timestamp
}