reliability<-function(trainpath, distance, kernel, estimatehyp, accuracythreshold){
    #COLLECT FULL LENGTH ACCURACY
    cv = crossvalidation(trainpath, distance, kernel, estimatehyp)
    accuracy = cv$accuracies[[length(cv$accuracies)]]
    #We remove the classes that have an accuracy lower than 1/|C|
    accuracy[which(accuracy<1/length(accuracy))]<-0
    numclus<-length(accuracy)

    #CALCULATE LEVEL OF ACCURACY (% of the total accuracy)
    acc<-accuracy*accuracythreshold/100

    #CALCULATE RELIABILITY 1 
    rel1<-reliability1(cv$accuracies, acc)
    #EXTRACT THRESHOLDS FOR RELIABILITY 2
    rel2<-reliability2(cv$probabilities, rel1, numclus)

    rel = list(rel1, rel2)
    return(rel)
}
