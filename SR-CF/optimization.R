optimization = function(probabilities, optimizer='ga',
    alpha=0.8, sr, reg='none', lambda=0, np, seed=0) {
    
    regret = 1      # costs for mis-classification

    set.seed(seed)
    ordervariables<-paste("V",c(1:(dim(probabilities[[1]])[2]-1)),sep="")
    cl = sort(unique(sapply(probabilities, function(i) { i$class[1] })))

    sigmainit<-runif(3, min=-1, max=1)
    if (sr$name == 'sr2') {
        sigmainit = runif(length(cl)+1, min=-1, max=1)
    }

    #Complete cost function
    cost_uniclass = function(sigma, probabilities, alpha, regret, reg, lambda){
        ce<-0
        ca<-0
        
        for(i in c(1:length(probabilities))){
            #Probabilities is a table, where each row corresponds to a time and we 
            #have the following information p1,p2,...,pk,t
            probs<-probabilities[[i]]
            trueclasses<-unique(probs$class)
            probs$class<-NULL
            probs<-probs[,ordervariables]
            
            #For all the probabilities obtained for a series (at each timestamp, one set), we extract the first one in
            #which s HALTS
            mat = t(probs)
            for (ti in 1:20)
                if (sr$func(mat[,ti], ti/20, sigma)==1)
                    break;
            
            #Add the cost of this series to the total expected cost
            ce = ce + ti/20
            predictedclass<-as.numeric(which.max(probs[ti,]))
            if(predictedclass != trueclasses[1])
                ca = ca + regret
        }
        switch(reg,
            'L0' = alpha*ca+(1-alpha)*ce+lambda*(sum(sigma!=0)),
            'L1' = alpha*ca+(1-alpha)*ce+lambda*(sum(abs(sigma))),
            alpha*ca+(1-alpha)*ce # default: no regularization
        )
    }

    switch(optimizer,
        'optim' = {
            #Basic optimization with optim
            result<-optim(sigmainit,cost_uniclass, probabilities=probabilities, alpha=alpha, regret=regret,
                reg=reg, lambda=lambda)
            sigma = result$par
        },
        'sa' = {
            #Simulated Annealing
            suppressMessages(library("GenSA"))
            lower<-c(1:length(sigmainit))*0-1
            upper<-c(1:length(sigmainit))*0+1
            result<-GenSA(sigmainit,cost_uniclass, probabilities=probabilities,
                alpha=alpha, regret=regret, reg=reg, lambda=lambda, upper=upper,
                lower=lower, control=c(maxit=100, verbose=TRUE))
            sigma = result$par
        },
        'ga' = {
            #Genetic algorithm (Used in the paper)
            suppressMessages(library("GA"))
            cost_uniclass2<-function(sigma, probabilities, alpha, regret, reg, lambda){
              c<- -1*cost_uniclass(sigma,probabilities,alpha,regret, reg, lambda)
            }
            lower<-c(1:length(sigmainit))*0-1
            upper<-c(1:length(sigmainit))*0+1
            result<-ga(type="real-valued", parallel=np,
                       cost_uniclass2, probabilities=probabilities, alpha=alpha,
                       regret=regret, reg=reg, lambda=lambda, lower=lower, upper=upper, seed=seed)
            sigma = result@solution[1,]
        },
        'pso' = {
            #Particle swarm optimization
            suppressMessages(library("pso"))
            result<-psoptim(sigmainit,cost_uniclass, probabilities=probabilities,
                alpha=alpha, regret=regret, reg=reg, lambda=lambda, control=c(maxit=100))
            sigma = result$par
        })

    return(sigma)
}
