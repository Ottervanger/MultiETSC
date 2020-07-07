GP<-function(traindata,trainclass,testdata,testclass,kernel,estimatehyp, theta=rep(1.0, ncol(traindata))){
  #CHANGE SOME PARAMETERS
  #Convergence threshold
  Thresh <- 1e-8;

  #Maximum iterations
  max.train.iter <- 24;
  
  #TRAIN the GP model
  model <- vbmp(traindata, trainclass, testdata,testclass,theta,
           control=list(bThetaEstimate=estimatehyp,bPlotFitting=F,
          maxIts=max.train.iter,sKernelType=kernel, Thresh=Thresh, nSampsIS=100));
  return(model)
}