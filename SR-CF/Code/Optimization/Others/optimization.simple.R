


#####EXECUTION#####

numbd<-4


numrep <- 4 #Number of repetition (30 in paper)
costtype <- 1 #for 0-1 accuracy cost
alg <- 3 # For genetic algorithm
alpha <- 0.8

regtype <- 1 # For L0
#regtype <- 2 # For L1
regparams <- c(0)  # For no regularization
#c(0,0.1,0.5 ,1,5 ,10 ,50) #For regularization

source("sources.R")


load(paste("../../../Databases/UCR-",numbd,".RData",sep=""))
cl<-as.numeric(levels(as.factor(database[[2]])))

for(lambda in regparams){

for(rep in 1:numrep){

sigmainit<-runif(length(cl)+1, min=-1, max=1)

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
result<-optim(sigmainit,cost_uniclass, probabilities=probabilities, 
              alpha=alpha, regret=regret,cl=cl, lambda=lambda)
}else if(alg==2){
#SImulated Annealing
library("GenSA")
lower<-c(1:length(sigmainit))*0-1
upper<-c(1:length(sigmainit))*0+1
result<-GenSA(sigmainit,cost_uniclass, probabilities=probabilities, alpha=alpha, 
              regret=regret, cl=cl,lambda=lambda,
              upper=upper, lower=lower, control=c(maxit=100, verbose=TRUE))
}else if(alg==3){
#Genetic algorithm (which appears in the paper)
library("GA")
cost_uniclass2<-function(sigma, probabilities, alpha, regret,cl, lambda){
  c<- -1*cost_uniclass(sigma,probabilities,alpha,regret,cl, lambda)
}
lower<-c(1:length(sigmainit))*0-1
upper<-c(1:length(sigmainit))*0+1
result<-ga(type = "real-valued",
   cost_uniclass2, probabilities=probabilities, alpha=alpha, regret=regret,cl=cl, lambda=lambda,
   min=lower, max=upper,seed=rep)
}else{
#PSO
library("pso")
result<-psoptim(sigmainit,cost_uniclass, probabilities=probabilities, alpha=alpha,
                regret=regret, cl=cl, lambda=lambda, control=c(trace=1, maxit=100))
}

fichero<-paste("../../../Results/Rules/Others/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",rep,"-",regtype,"-",lambda,".RData",sep="")
save(result,file=fichero)

}
}

source("selectsimple.R")
