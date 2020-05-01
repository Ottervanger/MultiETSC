costtype<-1
reg <- 0 # For no regularization
regtype<-1 # For L0 regularization
alg <- 3
numbd <-4
alpha <- 0.8



load(paste("../../../Databases/UCR-",numbd,".RData",sep=""))
cl<-as.numeric(levels(as.factor(database[[2]])))

#Simple linear rule. Only one for all classes.
s<-function(probabilities, sigma){
  rule<-sum(unlist(c(sort(probabilities[1:(length(probabilities)-1)]), probabilities[length(probabilities)]))*sigma)
  prediction<-(rule>0)
  return(prediction)
}

#Load predicted probabilities
load(paste("../../../Results/Probabilities/Test/Formatted/probs-",numbd,".RData",sep=""))
ordervariables<-paste("V",c(1:(dim(probabilities[[1]])[2]-1)),sep="")

#load rule
if(reg ==
   
   
   0){
load(paste("../../../Results/Rules/Others/NoReg/simpleresult-",numbd,"-",alg,"-",costtype,"-", alpha,"-",1,".RData",sep=""))
}else{
load(paste("../../../Results/Rules/Others/Reg/simpleresult-",numbd,"-",alg,"-",costtype,"-", alpha,"-",regtype,".RData",sep=""))
}
  
#Define parameters
if(alg==3){
  params<-result@solution[1,]
}else{
  params<-result$par
}

predictedclass<-c(1:dim(probabilities[[1]])[1])*NA
times<-c(1:dim(probabilities[[1]])[1])*NA
trueclasses<-c(1:dim(probabilities[[1]])[1])*NA

for(i in c(1:length(probabilities))){
  #Probabilities is a table, where each row corresponds to a time and we 
  #have the following information p1,p2,...,pk,t
  probs<-probabilities[[i]]
  trueclasses[i]<-unique(probs$class)
  probs$class<-NULL
  probs<-probs[,ordervariables]
  
  #Calculate rule value for all earlyness stamps and choose first t
  #in which it is fulfilled
  predictions<-apply(cbind(probs,c(1:20)/20),1,s,params)
  t<-which(predictions==1)[1]
  if(is.na(t)){predictedclass[i]<-as.numeric(which.max(probs[20,])); times[i]<-1
  }else{
  t<-t/20
  #Calculate predicted class
  predictedclass[i]<-as.numeric(which.max(probs[t*20,]))
#  if(dim(probs)[2]==2){
    predictedclass[i]<-cl[predictedclass[i]]
 # }
  times[i]<-t
  }
}


#Calculate mean accuracy
accuracy<-length(which(predictedclass==trueclasses))/length(probabilities)*100
#Calculate mean earlyness
times<-times[complete.cases(times)]
earlyness<-mean(times*5*20)

numpredicted<-length(!is.na(predictedclass))/length(predictedclass)*100

if(reg==0){
setwd("../../../Results/FinalResults/Others/NoReg")
}else{
setwd("../../../Results/FinalResults/Others/Reg")
}

fichero<-paste(getwd(),"/simpleresults-",numbd,"-",alg,"-",costtype,"-",alpha,"-", regtype,".txt",sep="")
write.table(c(accuracy,earlyness,numpredicted),file=fichero)

#Save predictions
fichero<-paste(getwd(),"/simplepredictions-",numbd,"-",alg,"-",costtype,"-",alpha,"-", regtype,".txt",sep="")
write.table(predictedclass,file=fichero)#Save predictions

#Saves times
fichero<-paste(getwd(),"/simpletimes-",numbd,"-",alg,"-",costtype,"-",alpha,"-", regtype,".txt",sep="")
write.table(times,file=fichero)#Save predictions

  

