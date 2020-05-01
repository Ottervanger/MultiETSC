`predictCPP` <-
function(obj, X.TEST=NULL) {
   ## Computes the predictive posteriors as defined in Section 4.5 of the paper.
   ## first two equations at page 1798
   con <- list(InfoLevel=0, sFILE.TRACE=NULL, bThetaEstimate=FALSE, 
        sKernelType="gauss", maxIts=50, Thresh=1e-4, tmpSave=NULL, nNodesQuad=49,
        nSampsTG=1000, nSampsIS=1000, nSmallNo=1e-10, parGammaSigma=1e-6, 
        parGammaTau=1e-6, bMonitor=FALSE, bPlotFitting=FALSE, method="quadrature");
   con[names(obj$con)] <- obj$con;
   if (con$method == "quadrature") {
      Nsamps  <- con$nNodesQuad;   
      genCPP  <- genCPP.quad;
   } else {
      Nsamps  <- con$nSampsTG;      
      genCPP  <- genCPP.classic;
   }
   X  <- obj$X;
   if (length(X.TEST)==0) X.TEST <- X;
   Y <- obj$Y;
   M <- obj$M;
   Kc <- ncol(Y);        ## Identify the number of classes
   N  <- nrow(X); Kd <- ncol(X);   ## Get number of samples and dimension of data
   theta <- covParams(obj);
   Ntest <- nrow(X.TEST);     ## Number of test points
   invPHI <- obj$invPHI;
   PHItest <- computeKernel(X, X.TEST, con$sKernelType, theta);
   PHItestSelf <- computeKernel(X.TEST, X.TEST, con$sKernelType, theta);
   Res <- t(crossprod(Y, invPHI)%*%PHItest);
   S <- (diag(PHItestSelf) - diag(crossprod(PHItest, invPHI)%*%PHItest));
   if (Kc > 2) { 
      Ptest <- genCPP(Ntest, Kc, Nsamps, Res, S);
   } else {
     Ptest <- genCPP.binary(Ntest, Kc, Res, S);
   }
   Ptest;
}

