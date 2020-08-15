crossvalidation = function(trainpath, cachepath, distance, kernel, estimatehyp) {
    cachename = paste('crossvalidation', kernel, estimatehyp, seed, sep='-')
    cachename = paste(cachepath, cachename, '.rds',  sep='')

    if (file.exists(cachename))
        return(readRDS(cachename))

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
    if (num_cores) {
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

    # function to loop over
    loop = function(repfold) {
        rep = repfold$rep
        fold = repfold$fold
        # we need to set a unique seed for each thread
        set.seed(seed+rep+reps*fold+reps*folds*repfold$earliness)
        #CREATE DATASETS for this repetition and fold
        traincv = train[-cv[[rep]][[fold]],]
        testcv = train[cv[[rep]][[fold]],]
        trainclasscv = trainclass[-cv[[rep]][[fold]]]
        testclasscv = trainclass[cv[[rep]][[fold]]]

        #CALCULATE early timestamp
        timestamp = (dim(train)[2])*repfold$earliness/nClassifiers

        #CALCULATE DISTANCE MATRIX for given early timestamp.
        DMtrain = distanceMatrix(train=traincv, earlyness=timestamp, distance=distance)
        DMtest = distanceMatrix(train=traincv, test=testcv, earlyness=timestamp, distance=distance)

        #TRAIN the GP . Hyperparameter estimation set to true
        model = GP(DMtrain,trainclasscv,DMtest,testclasscv,kernel,estimatehyp)

        #EXTRACT the accuracy for each class
        accaux = obtainaccuracy(model, testclasscv)

        #EXTRACT  of the posterior probabilities of the correctly classified series
        correct = which(predClass(model,numclus)==testclasscv)
        probabilitiesaux = matrix(nrow=length(correct),ncol=numclus+1)
        probabilitiesaux[,(1:numclus)] = model$Ptest[correct,]
        probabilitiesaux[,(numclus+1)] = testclasscv[correct]

        return(list(acc=accaux, prob=probabilitiesaux, earliness=repfold$earliness))
    }

    # run parallel loop
    if (num_cores)
        res = parLapply(cl, repList, loop)
    else
        res = lapply(repList, loop)
    # reducing the list of results into the expected data structure
    reducer = function(c,e) {
        c[['accuracies']][[e$earliness]]=rbind(c[['accuracies']][[e$earliness]], e$acc)
        c[['probabilities']][[e$earliness]]=rbind(c[['probabilities']][[e$earliness]], e$prob)
        c
    }
    ret = Reduce(reducer, res, list(accuracies=vector('list',nClassifiers), probabilities=vector('list',nClassifiers)))
    ret$accuracies = lapply(ret$accuracies, function(x) colMeans(x, na.rm=T))
    # save to cache
    saveRDS(ret, file=cachename, compress=FALSE)
    return(ret)
}





