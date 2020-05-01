GP<-function(traindata,trainclass,testdata,testclass,kernel,estimatehyp){
  
  ###DEFINE KERNEL
  if(kernel==1){
    sKernelType <- "iprod";
  }
  else if(kernel==2){
    sKernelType <- "gauss";
  }
  else if(kernel==3){
    sKernelType <- "cauchy";
  }
  else{
    sKernelType <- "laplace";
  }
  
  #CHANGE SOME PARAMETERS
  #Convergence threshold
  Thresh <- 1e-8;
  #Hyperparameter initial value
  theta <- rep(1.0, ncol(traindata));
  #Maximum iterations
  max.train.iter <- 24;
  
  #TRAIN the GP model
  model <- vbmp(traindata, trainclass, testdata,testclass,theta, 
           control=list(bThetaEstimate=estimatehyp,bPlotFitting=F, 
          maxIts=max.train.iter,sKernelType=sKernelType, Thresh=Thresh));
  
  return(model)
}