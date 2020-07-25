distanceMatrix = function(train, test, earlyness, distance) {
    if (distance != 1)
        stop('Only relGP with Euclidean distance is implemented')
    if (class(train) != 'matrix')
        train = unname(as.matrix(train))
    if (missing(test))
        test = train
    else if (class(test) != 'matrix')
        test = unname(as.matrix(test))
    dSqrd = distSqrd(train[,1:earlyness,drop=F],test[,1:earlyness,drop=F])
    dSqrd[which(dSqrd < 0)] = 0
    t(sqrt(dSqrd))
}
