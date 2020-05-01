
###This function select the rule with the median cost, from the 30
### executions of the genetic algorithm

alg<-3

for(costtype in c(1:1)){
  
for(numbd in c(4:4)){

alpha <- 0.8

results<-c(1:numrep)*NA

for(rep in c(1:numrep)){
  
load(paste("../../../Results/Rules/SR1-CF1/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",rep,".RData",sep=""))

results[rep]<-result@fitnessValue

}

aux<-which.min(abs(results-median(results)))
load(paste("../../../Results/Rules/SR1-CF1/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",aux,".RData",sep=""))
save(result,file=paste("../../../Results/Rules/SR1-CF1/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,".RData",sep=""))
}
}

