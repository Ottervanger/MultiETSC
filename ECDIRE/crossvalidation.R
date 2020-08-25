crossvalidation = function(trainpath, distance, kernel, estimatehyp) {
    #LOAD DATA
    data = loadData(trainpath)
    trainclass = data$class
    train = data$ts

    
    set.seed(seed)
    reps = 3
    folds = 5
    nClassifiers = 20
    #CREATE the sets for cross validation.
    cv = generateCVRuns(as.numeric(trainclass),ntimes=reps,nfold=folds,stratified=TRUE)

    # core cluster
    if (num_cores > 1) {
        cl = makeCluster(num_cores, type="FORK")
        on.exit(stopCluster(cl))
    }
    # list for the parallel loop
    repList = list()
    for(earliness in c(1:nClassifiers)) {
        for(rep in 1:reps) {
            for(fold in 1:folds) {
                repList = append(repList,list(list(rep=rep, fold=fold, earliness=earliness)))
            }
        }
    }

    # PRECALCULATE distance matrices for each timestep
    computeDistanceMatrix = function (earliness) {
        distanceMatrix(train=train, earlyness=(dim(train)[2])*earliness/nClassifiers, distance=distance)
    }

    # function to loop over
    loop = function(repfold) {
        rep = repfold$rep
        fold = repfold$fold
        # we need to set a unique seed for each thread
        set.seed(seed+rep+reps*fold+reps*folds*repfold$earliness)
        idxTest = cv[[rep]][[fold]]
        idxTrain = -idxTest

        #TRAIN the GP.
        model = GP(distanceMatrices[[repfold$earliness]][idxTrain,idxTrain], trainclass[idxTrain],
                   distanceMatrices[[repfold$earliness]][idxTest,idxTrain] , trainclass[idxTest],
                   kernel,estimatehyp)

        #EXTRACT the accuracy for each class
        accaux = classAccuracy(model, trainclass[idxTest])

        #EXTRACT  of the posterior probabilities of the correctly classified series
        correct = which(predClass(model,numclus)==trainclass[idxTest])
        probabilitiesaux = matrix(nrow=length(correct),ncol=numclus+1)
        probabilitiesaux[,(1:numclus)] = model$Ptest[correct,]
        probabilitiesaux[,(numclus+1)] = trainclass[idxTest][correct]

        return(list(acc=accaux, prob=probabilitiesaux, earliness=repfold$earliness))
    }

    # run parallel loop
    if (num_cores > 1) {
        distanceMatrices = parLapply(cl, c(1:nClassifiers), computeDistanceMatrix)
        res = parLapply(cl, repList, loop)
    } else {
        distanceMatrices = lapply(c(1:nClassifiers), computeDistanceMatrix)
        res = lapply(repList, loop)
    }
    assign("distanceMatrices", distanceMatrices, envir = .GlobalEnv)
    # reducing the list of results into the expected data structure
    reducer = function(c,e) {
        c[['accuracies']][[e$earliness]]=rbind(c[['accuracies']][[e$earliness]], e$acc)
        c[['probabilities']][[e$earliness]]=rbind(c[['probabilities']][[e$earliness]], e$prob)
        c
    }
    ret = Reduce(reducer, res, list(accuracies=vector('list',nClassifiers), probabilities=vector('list',nClassifiers)))
    ret$accuracies = lapply(ret$accuracies, function(x) colMeans(x, na.rm=T))
    return(ret)
}





