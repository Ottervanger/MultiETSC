reliability<-function(trainpath, distance, kernel, estimatehyp, accuracythreshold){

    cachepath = paste(".cache/rel/", gsub("^.*/([^/]*)_.*$", "\\1", trainpath),
                      "-", distance, "-", kernel, "-", estimatehyp, "-", accuracythreshold, ".rds", sep="")

    # retrieve cache if its there
    if (file.exists(cachepath)) {
        return(readRDS(cachepath))
    }

    #CALCULATE RELIABILITY 1 
    rel1<-reliability1(databasename, numclus, acc)
    #EXTRACT THRESHOLDS FOR RELIABILITY 2
    rel2<-reliability2(databasename, numclus)

    rel = list(rel1, rel2)
    saveRDS(rel, file=cachepath, compress=FALSE)
    return(rel)
}
