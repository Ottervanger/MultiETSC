#This list opens the files with the posterior probabilities 
#for the training data and saves all the results for each database
#in a list.

#Each element in the list is a matrix that corresponds to one series in the database.
#Each row of the matrix corresponds to a different earlyness value.

numbd <- 4

  probabilities<-list()
  
  for(earlynessperc in c(1:20)*5){
    
    probs<-as.data.frame(matrix(nrow=0,ncol=0))
    for(rep in c(1:1)){
    for(fold in c(1:5)){
    fichero<-paste("../../Results/Probabilities/Train/Raw/probs-",numbd,"-",earlynessperc,"-",rep,"-",fold,".txt",sep="")
    aux<-read.table(fichero, header=TRUE, check.names=FALSE)
    probs<-rbind(probs,aux)
    }
    }
    
    probs<-probs[,order(names(probs))]
    
    for(numseries in c(1:dim(probs)[1])){
    
    if(earlynessperc==5){
      probabilities[[numseries]]<-as.data.frame(matrix(nrow=20,ncol=dim(probs)[2]))
      names(probabilities[[numseries]])<-names(probs)
    }
    
    probabilities[[numseries]][earlynessperc/5,]<-as.numeric(probs[numseries,])
    
    }
  
  }
  
fichero<-paste("../../Results/Probabilities/Train/Formatted/probs-",numbd,".RData",sep="")
save(probabilities,file=fichero)
  

    
    
  
