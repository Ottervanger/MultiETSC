
#distance==1 for euclidean
#distance==2 for DTW

distanceMatrix<-function(train, test, earlyness, distance){
  
  ########################################################
  ##########For distance matrix of training set###########
  ########################################################
  if(missing(test)){
    
      if(distance==1){
      #Truncate series accordingly
      truncatedtrain<-as.matrix(train[,1:earlyness])
      #Calculate distance matrix using TSdist package
      dtrain<-as.matrix(tsDatabaseDistances(truncatedtrain, method=distance,
                                            upper=TRUE, diag=TRUE))
      }else{
        stop('Only relGP with Euclidean distance is implemented')
      }
    return(dtrain)
    
    ########################################################
    ##########For distance matrix of testing set###########
    ########################################################
    }else{
      dtest<-matrix(nrow=dim(test)[1],ncol=dim(train)[1])
      
      for(k in c(1:dim(test)[1])){
        #Truncate test series accordingly
        truncatedtest<-as.numeric(test[k,1:earlyness])
        
        #FOR EUCLIDEAN DISTANCE
        if(distance==1){
          #Truncate training series accordingly
          truncatedtrain<-as.matrix(train[,1:earlyness])
          #Calculate distance matrix using TSdist package
          dtest[k,]<-apply(as.matrix(truncatedtrain), 1, tsDistances, 
                           x=truncatedtest, distance=distance)
    
        }else{
          stop('Only relGP with Euclidean distance is implemented') 
        }
      }
    return(dtest)
  }
  
}

 

  

