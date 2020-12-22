prediction = function(trainpath, testpath, distance, kernel,
    optimizer, alpha, sr, reg, lambda, np, seed) {

    # Stopping rules
    sr1 = function(pr, pt, sigma) {
        if (length(pr) == 2) {
            o = if (pr[1] > pr[2]) 1:2 else c(2,1)
        } else {
            m = which.max(pr)
            pr = c(pr[m], max(pr[-m]))
        }
        (sigma[1]*pr[1]+sigma[2]*(pr[1]-pr[2])+sigma[3]*pt) > 0
    }

    sr2 = function(pr, pt, sigma) {
        if (length(pr) == 2) {
            o = if (pr[1] < pr[2]) 1:2 else c(2,1)
        } else {
            o = order(pr)
        }
        sum(unlist(c(pr[o], pt))*sigma) > 0
    }

    rule = list()
    rule$func = get(sr)
    rule$name = sr

    #Load predicted probabilities
    source('probabilities.R')
    probabilities_l = getProbabilities(trainpath, testpath, distance=distance, kernel=kernel, np=np, seed=seed)
    probabilities = probabilities_l$test
    ordervariables = paste("V",c(1:(dim(probabilities[[1]])[2]-1)),sep="")
    cl = sort(unique(sapply(probabilities, function(i) { i$class[1] })))

    #load rule
    source('optimization.R')
    sigma = optimization(probabilities_l$train, optimizer=optimizer,
        alpha=alpha, sr=rule, reg=reg, lambda=lambda, np=np, seed=seed)

    predictedclass = c(1:dim(probabilities[[1]])[1])*NA
    times = c(1:dim(probabilities[[1]])[1])*NA
    trueclasses = c(1:dim(probabilities[[1]])[1])*NA

    for(i in c(1:length(probabilities))){
        #Probabilities is a table, where each row corresponds to a time and we 
        #have the following information p1,p2,...,pk,t
        probs = probabilities[[i]]
        trueclasses[i] = unique(probs$class)
        probs$class = NULL
        probs = probs[,ordervariables]
        
        #Calculate rule value for all earlyness stamps and choose first t
        #in which it is fulfilled
        mat = t(probs)
        for (ti in 1:20)
            if (rule$func(mat[,ti], ti/20, sigma)==1) 
                break;
        if(is.na(ti)) {
            predictedclass[i] = as.numeric(which.max(probs[20,]))
            times[i] = 1
        } else {
            #Calculate predicted class
            predictedclass[i] = cl[as.numeric(which.max(probs[ti,]))]
            times[i] = ti/20
        }
    }
    result = list()
    result$accuracy = length(which(predictedclass==trueclasses))/length(probabilities)
    result$earliness = mean(times[complete.cases(times)])
    return(result)
}
