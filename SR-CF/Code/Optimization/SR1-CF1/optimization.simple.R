


#####EXECUTION#####

numbd<-4
costtype <- 1 #For 0-1 accuracy cost function
alg <- 3 #For genetic algorithm
alpha <- 0.8
numrep <- 4 #Number of repetitions of the optimization algorithm (30 in paper)

source("sources.R")

load(paste("../../../Databases/UCR-",numbd,".RData",sep=""))
cl<-as.numeric(levels(as.factor(database[[2]])))

for(rep in c(1:numrep)){

sigmainit<-runif(3, min=-1, max=1)

#Read posterior probabilities for the given database
fichero<-paste("../../../Results/Probabilities/Train/Formatted/probs-",numbd,".RData",sep="") 
load(fichero)
ordervariables<-paste("V",c(1:(dim(probabilities[[1]])[2]-1)),sep="")

#Define different costs for mis-classification
#0-1 loss
if(costtype==1){
  regret<-1
}


if(alg==1){
  #Basic optimization with optim
  result<-optim(sigmainit,cost_uniclass, probabilities=probabilities, alpha=alpha, regret=regret,cl=cl)
}else if(alg==2){
  #SImulated Annealing
  library("GenSA")
  lower<-c(1:length(sigmainit))*0-1
  upper<-c(1:length(sigmainit))*0+1
  result<-GenSA(sigmainit,cost_uniclass, probabilities=probabilities, alpha=alpha, regret=regret, cl=cl,upper=upper, lower=lower, control=c(maxit=100, verbose=TRUE))
}else if(alg==3){
  #Genetic algorithm (Used in the paper)
  library("GA")
  cost_uniclass2<-function(sigma, probabilities, alpha, regret,cl){
    c<- -1*cost_uniclass(sigma,probabilities,alpha,regret,cl)
  }
  lower<-c(1:length(sigmainit))*0-1
  upper<-c(1:length(sigmainit))*0+1
  result<-ga(type = "real-valued",
             cost_uniclass2, probabilities=probabilities, alpha=alpha, regret=regret,cl=cl,min=lower, max=upper,seed=rep)
}else{
  #PSO
  library("pso")
  result<-psoptim(sigmainit,cost_uniclass, probabilities=probabilities, alpha=alpha, regret=regret, cl=cl, control=c(trace=1, maxit=100))
}

fichero<-paste("../../../Results/Rules/SR1-CF1/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",rep,".RData",sep="")
save(result,file=fichero)
}

source("selectsimple.R")

