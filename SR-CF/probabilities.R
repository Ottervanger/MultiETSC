getProbabilities = function(trainpath, testpath, distance, kernel='iprod', np, seed=0) {

    thetaestimate = FALSE
    nfold = 5
    nClassifiers = 20

    #Load sources
    options(rgl.useNULL=TRUE)
    suppressMessages(source("sources.R"))

    traindata = read.table(trainpath)
    train = traindata[,-1]
    classestrain = as.factor(traindata[,1])

    testdata = read.table(testpath)
    test = testdata[,-1]
    classestest = as.factor(testdata[,1])

    set.seed(seed)
    cv<-generateCVRuns(classestrain, ntimes=1, nfold=nfold, stratified=TRUE)

    itemList = list()
    for (fold in c(1:nfold)) {
        for (earliness in c(1:nClassifiers)) {
            itemList = append(itemList,list(list(fold=fold, earliness=earliness)))
        }
    }

    ret = list()

    fun = function(item) {
        set.seed(seed+item$fold*nClassifiers+item$earliness)
        idxTest = cv[[1]][[item$fold]]
        idxTrain = -idxTest
        predicted = trainmodel(
            distanceMatrices[[item$earliness]][idxTrain,idxTrain],
            classestrain[idxTrain],
            distanceMatrices[[item$earliness]][idxTest,idxTrain] ,
            classestrain[idxTest],
            kernel=kernel,
            thetaestimate=thetaestimate)
        predicted = as.data.frame(predicted)
        predicted$class = classestrain[idxTest]
        list(earliness=item$earliness, predicted=predicted)
    }

    # PRECALCULATE distance matrices for each timestep
    computeDistanceMatrix = function (earliness) {
        distanceMatrix(train=train, earlyness=earliness*dim(train)[2]/nClassifiers, distance=distance)
    }

    if (np > 1) {
        cl = cl = makeCluster(np, type="FORK")
        on.exit(stopCluster(cl))
        distanceMatrices = parLapply(cl, c(1:nClassifiers), computeDistanceMatrix)
        res = parLapply(cl, itemList, fun)
    } else {
        distanceMatrices = lapply(c(1:nClassifiers), computeDistanceMatrix)
        res = lapply(itemList, fun)
    }
    set.seed(seed)
    pros = Reduce(function(r, e) {r[[e$earliness]]=rbind(r[[e$earliness]], e$predicted); r},
                       res, rep(list(data.frame()), nClassifiers))

    pnames = names(pros[[1]])
    probabilities = aperm(array(unlist(pros), dim=c(dim(pros[[1]]),nClassifiers)), c(3,2,1))
    probabilities = lapply(seq(dim(probabilities)[3]), function(x) { r = as.data.frame(probabilities[,,x]); names(r) = pnames; r[,order(names(r))]} )
    ret$train = probabilities

    funAll = function(earlynessperc) {
        distTest = distanceMatrix(train=train, test=test, earlyness=earlynessperc*dim(train)[2]/nClassifiers, distance=distance)
        predicted = trainmodel(
            distanceMatrices[[earlynessperc]],
            classestrain,
            distTest,
            classestest,
            kernel=kernel,
            thetaestimate=thetaestimate)
        predicted = as.data.frame(predicted)
        predicted$class = classestest
        as.data.frame(predicted)
    }
    if (np > 1) {
        pros = parLapply(cl, c(1:nClassifiers), funAll)
    } else {
        pros = lapply(c(1:nClassifiers), funAll)
    }
    pnames = names(pros[[1]])
    probabilities = aperm(array(unlist(pros), dim=c(dim(pros[[1]]),nClassifiers)), c(3,2,1))
    probabilities = lapply(seq(dim(probabilities)[3]), function(x) { r = as.data.frame(probabilities[,,x]); names(r) = pnames; r[,order(names(r))]} )
    ret$test = probabilities
    return(ret)
}
