distanceMatrix = function(train, test, earlyness, distance) {
    if (distance != 1)
        stop('Only relGP with Euclidean distance is implemented')
    if (class(train) != 'matrix')
        train = unname(as.matrix(train))
    if (missing(test))
        return(t(sqrt(distSqrd(train[,1:earlyness,drop=F],train[,1:earlyness,drop=F]))))
    if (class(test) != 'matrix')
        test = unname(as.matrix(test))
    t(sqrt(distSqrd(train[,1:earlyness,drop=F],test[,1:earlyness,drop=F])))
}

