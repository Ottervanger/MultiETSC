reliability<-function(trainpath, distance, kernel, estimatehyp, accuracythreshold){
    # GO-TODO: this path only works for original UCR filenames
    cachepath = paste(".cache/", gsub("^.*/([^/]*/[^/]*).*\\..*$", "\\1", trainpath), "/", sep="")
    cachename = paste('reliability', kernel, estimatehyp, accuracythreshold, seed, sep='-')
    cachename = paste(cachepath, cachename, '.rds', sep="")
    
    # check cache integrety
    if (system(sprintf('sha1sum -c %s.sha1', cachepath), ignore.stdout=T, ignore.stderr=T) == 0) {
        if (file.exists(cachename)) {
            return(readRDS(cachename))
        }
    } else {
        # if not valid, reinit
        unlink(cachepath, recursive=T)
        dir.create(cachepath, showWarnings=F, recursive=T)
        system(sprintf('sha1sum %s > %s.sha1', trainpath, cachepath))
    }

    #COLLECT FULL LENGTH ACCURACY
    cv = crossvalidation(trainpath, cachepath, distance, kernel, estimatehyp)
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
    saveRDS(rel, file=cachename, compress=FALSE)
    return(rel)
}
