
#######################AUXILIARY FUNCTIONS FOR TRAINING SVM####################################

#Function to train GP models given a training set
trainmodel<-function(traindata,trainclass,testdata,testclass,kernel,thetaestimate){
  Thresh <- 1e-8;
  theta <- rep(1.0, ncol(traindata));
  max.train.iter <- 24;
  model <- vbmp(traindata, trainclass, testdata,testclass,theta, 
                control=list(bThetaEstimate=thetaestimate,bPlotFitting=F, 
                             maxIts=max.train.iter,sKernelType=kernel, Thresh=Thresh));
  
  probs<-model$Ptest
  
  return(probs)
}



