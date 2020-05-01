earlyDP<-function(x, y, distance, param){
  
  tamx<-length(x)
  tamy<-length(y)
  
  #We calculate the whole cost matrix
  if(distance=="dtw"){
    if(param>=tamx){costMatrix<-dtwDistance(x,y)
    }else{costMatrix<-dtwDistance(x,y, sigma=param)}
    costMatrix<-matrix(costMatrix, nrow=tamx, ncol=tamy, byrow=TRUE)
  }else{
    costMatrix<-edrDistance(x,y, epsilon=param)
    costMatrix<-matrix(costMatrix, nrow=tamx+1, ncol=tamy+1, byrow=TRUE)
    costMatrix <- costMatrix[2:dim(costMatrix)[1],2:dim(costMatrix)[2]]
  }
  
  #We convert the costMatrix into a matrix (it is iniatially defined as a vector)
  
  
  if(distance=="dtw" && param<tamx){
  #We select the candidates
  candidates<-costMatrix[tamx,((tamy-param):tamy)]
  #We calculate the symmetric normalization constants
  normalizationconstants<-c((tamy-param):tamy)+tamx
  }else{
  #We select the candidates
  candidates<-costMatrix[tamx,]
  #We calculate the symmetric normalization constants
  normalizationconstants<-c(1:tamy)+tamx  
  }
  #We calculate the normalized candidates
  normalizedcandidates<-candidates/normalizationconstants
  
  #The minimum distance is chosen.
  d<-min(normalizedcandidates)
  return(d)
}