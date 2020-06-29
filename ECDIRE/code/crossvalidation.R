crossvalidation = function(trainpath, cachepath, distance, kernel, estimatehyp) {
    cachename = paste('crossvalidation', distance, kernel, estimatehyp, sep='-')
    cachename = paste(cachepath, cachename, '.rds',  sep='')

    # GO-TODO: set globally?
    seed = 100

    # retrieve cache if its there
    if (file.exists(cachename)) {
        return(readRDS(cachename))
    }

    # create dir if not there yet
    dir.create(dirname(cachename), recursive=TRUE, showWarnings=FALSE)

    #LOAD DATA
    data = loadData(trainpath)
    trainclass = data$class
    train = data$ts

    #CREATE the sets for the 10 times repeat 5-fold cross validation.
    set.seed(seed)
    reps = 10
    folds = 5
    cv = generateCVRuns(as.numeric(trainclass),ntimes=reps,nfold=folds,stratified=TRUE)
    require("parallel")

    #Initilize results
    accuracies = list()
    probabilities = list()

    # core cluster
    cl = makeCluster(detectCores(), type="FORK")
    on.exit(stopCluster(cl))
    # list for the parallel loop
    repsXfolds = list()
    for(rep in 1:reps) {
        for(fold in 1:folds) {
            repsXfolds = append(repsXfolds,list(list(rep=rep, fold=fold)))
        }
    }

    # function to loop over
    loop = function(repfold) {
        rep = repfold$rep
        fold = repfold$fold
        # we need to set a unique seed for each thread
        set.seed(seed+rep+reps*fold)
        #CREATE DATASETS for this repetition and fold
        traincv = train[-cv[[rep]][[fold]],]
        testcv = train[cv[[rep]][[fold]],]
        trainclasscv = trainclass[-cv[[rep]][[fold]]]
        testclasscv = trainclass[cv[[rep]][[fold]]]

        #CALCULATE early timestamp
        earlyness = earlynessperc*(dim(train)[2])/100

        #CALCULATE DISTANCE MATRIX for given early timestamp.
        DMtrain = distanceMatrix(train=traincv, earlyness=earlyness, distance=distance)
        DMtest = distanceMatrix(train=traincv, test=testcv, earlyness=earlyness, distance=distance)

        #TRAIN the GP . Hyperparameter estimation set to true
        model = GP(DMtrain,trainclasscv,DMtest,testclasscv,kernel,estimatehyp)

        #EXTRACT the accuracy for each class
        accaux = obtainaccuracy(model, testclasscv)

        #EXTRACT  of the posterior probabilities of the correctly classified series
        correct = which(predClass(model,numclus)==testclasscv)
        probabilitiesaux = matrix(nrow=length(correct),ncol=numclus+1)
        probabilitiesaux[,(1:numclus)] = model$Ptest[correct,]
        probabilitiesaux[,(numclus+1)] = testclasscv[correct]

        return(list(acc=accaux, prob=probabilitiesaux))
    }

    for(earlynessperc in c(1:20)*5) {
        clusterExport(cl, "earlynessperc", env=environment())
        res = parLapply(cl, repsXfolds, loop)
        accprob = Reduce(function(c,e) list(acc=rbind(c$acc, e$acc), prob=rbind(c$prob, e$prob)), res)
        accuracies[[earlynessperc]] = accprob$acc
        probabilities[[earlynessperc]] = accprob$prob
    }
    ret = list(accuracies=accuracies, probabilities=probabilities)
    # save to cache
    saveRDS(ret, file=cachename, compress=FALSE)
    return(ret)
}


