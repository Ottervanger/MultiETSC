getProbabilities = function(trainpath, testpath, cachepath,
    distance='euclidean', kernel='iprod', np, seed=0) {

    # parameter passed to distance function with changing interpretation
    # depending on the distance metric eg. dtw window, edit distance threshold, 
    # number of Fourier coefficients considered for Fourier distance.
    param = 5 
    thetaestimate = FALSE
    nfold = 5
    nClassifiers = 20
    
    cachename = paste('probabilities', distance, kernel, seed, sep='-')
    cachename = paste(cachepath, cachename, '.rds', sep="")

    if (file.exists(cachename))
        return(readRDS(cachename))

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
        set.seed(seed*1000+item$fold*nClassifiers+item$earliness)
        traincv = train[-cv[[1]][[item$fold]],]
        testcv = train[cv[[1]][[item$fold]],]
        trainclasscv = classestrain[-cv[[1]][[item$fold]]]
        testclasscv = classestrain[cv[[1]][[item$fold]]]
        predicted = trainmodel(
            traincv,
            trainclasscv,
            testcv,
            testclasscv,
            kernel=kernel,
            earlyness=round(item$earliness*dim(traincv)[2]/nClassifiers),
            distance=distance,
            param=param,
            thetaestimate=thetaestimate)
        predicted = as.data.frame(predicted)
        predicted$class = testclasscv
        list(earliness=item$earliness, predicted=predicted)
    }

    if (np >1) {
        cl = cl = makeCluster(np, type="FORK")
        on.exit(stopCluster(cl))
        res = parLapply(cl, itemList, fun)
    } else {
        res = lapply(itemList, fun)
    }
    set.seed(seed)
    pros = Reduce(function(r, e) {r[[e$earliness]]=rbind(r[[e$earliness]], e$predicted); r},
                       res, rep(list(data.frame()), nClassifiers))

    pnames = names(pros[[1]])
    probabilities = aperm(array(unlist(pros), dim=c(dim(pros[[1]]),nClassifiers)), c(3,2,1))
    probabilities = lapply(seq(dim(probabilities)[3]), function(x) { r = as.data.frame(probabilities[,,x]); names(r) = pnames; r[,order(names(r))]} )
    ret$train = probabilities

    pros = rep(list(data.frame()), nClassifiers)
    for (earlynessperc in c(1:nClassifiers)) {
        predicted = trainmodel(
            train,
            classestrain,
            test,
            classestest,
            kernel=kernel,
            earlyness=round(earlynessperc*dim(train)[2]/nClassifiers),
            distance=distance,
            param=param,
            thetaestimate=thetaestimate)
        predicted = as.data.frame(predicted)
        predicted$class = classestest

        pros[[earlynessperc]] = rbind(pros[[earlynessperc]], predicted)
    }

    pnames = names(pros[[1]])
    probabilities = aperm(array(unlist(pros), dim=c(dim(pros[[1]]),nClassifiers)), c(3,2,1))
    probabilities = lapply(seq(dim(probabilities)[3]), function(x) { r = as.data.frame(probabilities[,,x]); names(r) = pnames; r[,order(names(r))]} )
    ret$test = probabilities

    saveRDS(ret, file=cachename, compress=FALSE)
    return(ret)
}
