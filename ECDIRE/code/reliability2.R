#This function calculates the thresholds for the 2nd level 
#reliability.

reliability2<-function(databasename, numclus){
  
  thresholds<-matrix(nrow=20,ncol=numclus)
  
    #For each early timestamp
    for(earlyness in c(1:20)*5){
      
      #Collect posterior probabilities of the cross validation process
      file<-paste(getwd(),"/results/probabilities/prob-",databasename,"-",earlyness,".txt",sep="")
      #Read the data and some type modifications
      aux<-read.table(file,stringsAsFactors =FALSE)
      errores<-which(aux[,1]=="V1")
      if(length(errores)!=0){aux<-aux[-errores,]}
      if(is.character(aux[,1])){
        aux<-as.data.frame(sapply(aux, as.numeric))
      }
      #Read first level reliability
      fichero<-paste(getwd(),"/results/reliabilities/rel1-",databasename,".txt",sep="")
      rel1<-as.numeric(t(read.table(fichero)))
    
        #For each class that surpasses the accuracy limit
        for(clus in c(1:numclus)){
  
        if(is.na(rel1[clus])==FALSE){
          
          #Calculate the thresholds
          aux2<-aux[aux[,dim(aux)[2]]==clus,]
          if(dim(aux2)[1]!=0){
          aux2[,dim(aux2)[2]]<-NULL
          #Difference between the posterior probabilities
          aux2<--(aux2-aux2[,clus])[,-clus]
          if(!is.null(dim(aux2)[1])){
            #Choose the minimum for each instance
            aux2<-apply(aux2,2,min)[1:(numclus-1)]}
            #Choose the minimum for each class
            thresholds[earlyness/5,clus]<-min(aux2) 
          }else{
          thresholds[earlyness/5,clus]<-1  
          }
        }else{
        thresholds[earlyness/5,clus]<-1 
        }
}
}
return(thresholds)
}
  
