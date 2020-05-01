

aux<-regparams*NA
cost<-c(1:10)*NA
i<-1
j<-1


for(lambda in c(regparams)){

results<-c(1:numrep)*NA

for(rep in c(1:numrep)){
load(paste("../../../Results/Rules/Others/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",rep,"-",regtype,"-",lambda,".RData",sep=""))

if(alg==3){
results[rep]<-result@fitnessValue
}else{
results[rep]<-result$value
}
}

#Para cada lamdba guardar la repetici?n con coste mediano
aux[i]<-which.min(abs(results-median(results)))


#Para cada lambda, cargar la repeticiÃ³n con el coste median
load(paste("../../../Results/Rules/Others/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",aux[i],"-",regtype,"-",lambda,".RData",sep=""))

if(lambda==0){
  save(result,file=paste("../../../Results/Rules/Others/NoReg/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-", regtype,".RData",sep=""))
  i<-i+1
}else{
#Calcular su coste, eliminado la parte de la regularizaciÃ³n
if(regtype==1){
  cost[j]<-result@fitnessValue+lambda*(sum(result@solution!=0))
  }
else{
  cost[j]<-result@fitnessValue+lambda*(sum(result@solution))
}
i<-i+1
j<-j+1
}
}

if(regparams!=0){
#First type of selection, minimum cost
aux2<-regparams[which.max(cost)]
aux1<-aux[which.max(cost)+1]
load(paste("../../../Results/Rules/Others/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-",aux1,"-", regtype,"-",aux2,".RData",sep=""))
save(result,file=paste("../../../Results/Rules/Others/Reg/simpleresult-",numbd,"-",alg,"-",costtype,"-",alpha,"-", regtype,".RData",sep=""))
}



