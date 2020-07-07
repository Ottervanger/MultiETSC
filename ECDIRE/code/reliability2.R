#This function calculates the thresholds for the 2nd level reliability.
reliability2 = function (probabilities, rel1, numclus) {
    thresholds = matrix(nrow=20,ncol=numclus)

    #For each early timestamp
    for (earliness in c(1:20)) { 
        #Collect posterior probabilities of the cross validation process and some type modifications
        aux = as.data.frame(probabilities[[earliness]])
        #Read first level reliability
        rel1 = as.numeric(t(as.data.frame(rel1)))
    
        #For each class that surpasses the accuracy limit
        for(clus in c(1:numclus)){
            if (is.na(rel1[clus]) == TRUE) {
                thresholds[earliness,clus] = 1
                next
            }
            #Calculate the thresholds
            aux2 = aux[aux[,dim(aux)[2]]==clus,]
            if (dim(aux2)[1] == 0) {
                thresholds[earliness,clus] = 1
                next
            }
            aux2[,dim(aux2)[2]] = NULL
            #Difference between the posterior probabilities
            aux2 = -(aux2-aux2[,clus])[,-clus]
            if(!is.null(dim(aux2)[1])){
                #Choose the minimum for each instance
                aux2 = apply(aux2,2,min)[1:(numclus-1)]
            }
            #Choose the minimum for each class
            thresholds[earliness,clus] = min(aux2)
        }
    }
    return(thresholds)
}
    
