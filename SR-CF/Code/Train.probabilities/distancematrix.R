
#distance==1 for euclidean
#distance==2 for DTW

distanceMatrix<-function(train, test, earlyness, distance, param){
  
  ########################################################
  ##########For distance matrix of training set###########
  ########################################################
  if(is.null(test)){
    truncatedtrain<-as.matrix(train[,1:earlyness])
    if(distance==1){
      dtrain<-as.matrix(TSDatabaseDistances(truncatedtrain, distance="euclidean"))

    }else if(distance==2){
      dtrain<-matrix(nrow=dim(train)[1], ncol=dim(train)[1])
      for(i in c(1:dim(train)[1])){
        for(j in c(1:dim(train)[1])){
          x<-as.numeric(train[i,])
          y<-as.numeric(train[j,])
          truncatedx<-x[1:earlyness]
          paramm<-c(round(seq(4,32,4)*dim(train)[2]/100),dim(train)[2])[param]
          truncatedy<-y[1:min((length(truncatedx)+paramm),length(y))]
          dtrain[i,j]<-earlyDP(truncatedx,truncatedy,"dtw",paramm)
        }}
    }else if(distance==3){
      dtrain<-matrix(nrow=dim(train)[1], ncol=dim(train)[1])
      for(i in c(1:dim(train)[1])){
        for(j in c(1:dim(train)[1])){
          x<-as.numeric(train[i,])
          y<-as.numeric(train[j,])
          truncatedx<-x[1:earlyness]
          paramm<-0.2*sd(as.matrix(train))*param
          dtrain[i,j]<-earlyDP(truncatedx,y,"edr",paramm)}}
    }else if(distance==5){
    paramm<-min(c(round(seq(4,32,4)*dim(train)[2]/100),dim(train)[2])[param], dim(truncatedtrain)[2])
    dtrain<-as.matrix(tsDatabaseDistances(truncatedtrain, method="fourier",
                                            n=paramm, upper=TRUE, diag=TRUE))  
    }else{
      paramm<-seq(mean(as.matrix(train))-sd(as.matrix(train)),mean(as.matrix(train))+sd(as.matrix(train)),0.4)[param]
      dtrain<-as.matrix(tsDatabaseDistances(truncatedtrain, method="tquest",
                                            tau=paramm, upper=TRUE, diag=TRUE))  
    }
    return(dtrain)
    
    ########################################################
    ##########For distance matrix of testing set###########
    ########################################################
    }else{
      dtest<-matrix(nrow=dim(test)[1],ncol=dim(train)[1])
      
      for(k in c(1:dim(test)[1])){
        
        truncatedtest<-as.numeric(test[k,1:earlyness])
        
        #FOR EUCLIDEAN DISTANCE SIMPLY TRUNCATE SERIES
        truncatedtrain<-as.matrix(train[,1:earlyness])
        if(distance==1){
          dtest[k,]<-apply(as.matrix(truncatedtrain), 1, TSDistances, 
                  x=truncatedtest, distance="euclidean")
       
          }else if(distance==2){
          
          for(i in c(1:dim(train)[1])){
            x<-as.numeric(test[k,])
            truncatedx<-truncatedtest
            y<-as.numeric(train[i,])
            paramm<-c(round(seq(4,32,4)*dim(train)[2]/100),dim(train)[2])[param]
            aux<-min(length(truncatedx)+paramm, length(y))
            truncatedy<-y[1:aux]
            dtest[k,i]<-earlyDP(truncatedx,truncatedy,"dtw",paramm)
          }
        }else if(distance==3){
          
          for(i in c(1:dim(train)[1])){
            x<-as.numeric(test[k,])
            truncatedx<-truncatedtest
            y<-as.numeric(train[i,])
            paramm<-0.2*sd(as.matrix(train))*param
            dtest[k,i]<-earlyDP(truncatedx,y,"edr",paramm) }
            
        }else if(distance==5){
          paramm<-min(c(round(seq(4,32,4)*dim(train)[2]/100),dim(train)[2])[param], dim(truncatedtrain)[2])
          dtest[k,]<-apply(as.matrix(truncatedtrain), 1, tsDistances, 
                  x=truncatedtest, distance="fourier", n=paramm)
        }else{
          paramm<-seq(mean(as.matrix(train))-sd(as.matrix(train)),mean(as.matrix(train))+sd(as.matrix(train)),0.4)[param]
          dtest[k,]<-apply(as.matrix(truncatedtrain), 1, tsDistances, 
                           x=truncatedtest, distance="tquest", tau=paramm)
        }
      }
    return(dtest)
  }
  
}

 

  

