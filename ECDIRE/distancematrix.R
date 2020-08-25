distanceMatrix<-function(train, test=NULL, earlyness, distance) {
    if (distance$metric == "euclidean") {
        if (class(train) != 'matrix')
            train = unname(as.matrix(train))
        if (missing(test))
            test = train
        else if (class(test) != 'matrix')
            test = unname(as.matrix(test))
        dSqrd = distSqrd(train[,1:earlyness,drop=F],test[,1:earlyness,drop=F])
        dSqrd[which(dSqrd < 0)] = 0
        return(t(sqrt(dSqrd)))
    }
    if (!missing(test))
        test = test[,1:earlyness,drop=F]

    distArgs = c(
        list(train[,1:earlyness,drop=F], test, distance=distance$metric),
        switch(distance$metric,
            "edr"       = list(epsilon=sd(as.matrix(train))*distance$sigma),
            "fourier"   = list(n=min(distance$n,earlyness/2)),
            "tquest"    = list(tau=mean(as.matrix(train))+sd(as.matrix(train))*distance$tau)
        )
    )
    if (distance$metric == "dtw" || distance$metric == "edr")
        return(t(as.matrix(do.call("TSDatabaseDistances", distArgs)))/2*earlyness)
    return(t(as.matrix(do.call("TSDatabaseDistances", distArgs))))
}
