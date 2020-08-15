
#######################AUXILIARY FUNCTIONS FOR TRAINING SVM####################################

#Function to train GP models given a training set
trainmodel<-function(traindata,trainclass,testdata,testclass,kernel, earlyness, distance, param,thetaestimate){
  
  #Calculate feature matrix
  DMtrain<-distanceMatrix(traindata, test=NULL, earlyness, distance, param)
  DMtest<-distanceMatrix(traindata, test=testdata, earlyness, distance, param)
  
  Thresh <- 1e-8;
  theta <- rep(1.0, ncol(DMtrain));
  max.train.iter <- 24;
  model <- vbmp(DMtrain, trainclass, DMtest,testclass,theta, 
                control=list(bThetaEstimate=thetaestimate,bPlotFitting=F, 
                             maxIts=max.train.iter,sKernelType=kernel, Thresh=Thresh));
  
  probs<-model$Ptest
  
  return(probs)
}



