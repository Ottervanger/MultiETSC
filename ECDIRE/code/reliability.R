reliability<-function(trainpath, distance, kernel, estimatehyp, accuracythreshold){
    # GO-TODO: this path only works for original UCR filenames
    cachepath = paste(".cache/", gsub("^.*/([^/]*)_.*$", "\\1", trainpath), "/", sep="")
    cachename = paste('reliability', distance, kernel, estimatehyp, accuracythreshold, sep='-')
    cachename = paste(cachepath, cachename, '.rds', sep="")

    #COLLECT FULL LENGTH ACCURACY
    cv = crossvalidation(trainpath, cachepath, distance, kernel, estimatehyp)
    accuracy = colMeans(as.data.frame(cv$accuracies[[100]]),na.rm=TRUE)
    accuracy[which(accuracy=="NaN")]<-0
    #We remove the classes that have an accuracy lower than 1/|C|
    accuracy[which(accuracy<1/length(accuracy))]<-0
    numclus<-length(accuracy)

    #CALCULATE LEVEL OF ACCURACY (% of the total accuracy)
    acc<-accuracy*accuracythreshold/100

    # retrieve cache if its there
    if (file.exists(cachename)) {
        return(readRDS(cachename))
    }

    #CALCULATE RELIABILITY 1 
    rel1<-reliability1(cv$accuracies, acc, numclus)
    #EXTRACT THRESHOLDS FOR RELIABILITY 2
    rel2<-reliability2(cv$probabilities, rel1, numclus)

    rel = list(rel1, rel2)
    saveRDS(rel, file=cachename, compress=FALSE)
    return(rel)
}
