prediction = function(trainpath, testpath, distance, kernel,
    optimizer, alpha, sr, reg, lambda, np, seed) {

    # Stopping rules
    sr1 = function(probabilities, sigma) {
        probabilities[1:(length(probabilities)-1)]<-sort(probabilities[1:length(probabilities)-1], decreasing=TRUE)
        rule<-sigma[1]*probabilities[1]+sigma[2]*(probabilities[1]-probabilities[2])+sigma[3]*probabilities[length(probabilities)]
        (rule>0)
    }

    sr2 = function(probabilities, sigma) {
        rule<-sum(unlist(c(sort(probabilities[1:(length(probabilities)-1)]), probabilities[length(probabilities)]))*sigma)
        (rule>0)
    }

    rule = list()
    rule$func = get(sr)
    rule$name = sr

    cachepath = paste(".cache/", gsub("^.*/([^/]*/[^/]*).*\\..*$", "\\1", trainpath), "/", sep="")

    # check cache integrety
    if (system(sprintf('sha1sum -c %s.sha1', cachepath), ignore.stdout=T, ignore.stderr=T) != 0) {
        # reinit
        unlink(cachepath, recursive=T)
        dir.create(cachepath, showWarnings=F, recursive=T)
        system(sprintf('sha1sum %s > %s.sha1', trainpath, cachepath))
    }

    #Load predicted probabilities
    source('probabilities.R')
    probabilities_l = getProbabilities(trainpath, testpath, cachepath, distance=distance, kernel=kernel, np=np, seed=seed)
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
        predictions = apply(cbind(probs,c(1:20)/20),1,sr,sigma)
        t = which(predictions==1)[1]
        if(is.na(t)) {
            predictedclass[i] = as.numeric(which.max(probs[20,]))
            times[i] = 1
        } else {
            t = t/20
            #Calculate predicted class
            predictedclass[i] = as.numeric(which.max(probs[t*20,]))
            predictedclass[i] = cl[predictedclass[i]]
            times[i] = t
        }
    }
    result = list()
    result$accuracy = length(which(predictedclass==trueclasses))/length(probabilities)
    result$earliness = mean(times[complete.cases(times)])
    return(result)
}
