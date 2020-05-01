`vbmp` <-
function(X, t.class, X.TEST, t.class.TEST, theta,  control = list()) {
## X - Feature matrix for parameter 'estimation' - of dimension N x Kd
## t.class - The corresponing target values - class labels
## X.TEST - Feature matrix to compute out-of-sample (test) prediction errors and likelihoods
## t.TEST - Corresponding target values for test data
## theta - The covariance function parameters - e.g. scaling coefficients for each dimension
## bThetaEstimate = if covariance parameter estimation switched on - (FALSE if switched off)
## maxIts - the maximum number of variational EM steps to take
## sKernelType - Select from Gaussian, Polynomial or Linear Inner product
## Thresh - Convergence threshold on marginal likelihood lower-bound
## InfoLevel - 0 to suppress tracing ( > 0  to print different levels
##             of monitoring information)
    # ------------ check data dimension
    if (is.null(dim(X)))  X <- matrix(X,nrow=length(X),ncol=1);
    if (is.null(dim(X.TEST))){
        X.TEST <- matrix(X.TEST, ncol=ncol(X));
    } else if (ncol(X.TEST) != ncol(X)) stop("Number of cols differ between X and X.TEST");

    # ------------ check parameters    
    con <- list(InfoLevel=0, sFILE.TRACE=NULL, bThetaEstimate=FALSE, 
        sKernelType="gauss", maxIts=50, Thresh=1e-4, tmpSave=NULL, nNodesQuad=49,
        nSampsTG=1000, nSampsIS=1000, nSmallNo=1e-10, parGammaSigma=1e-6, 
        parGammaTau=1e-6, bMonitor=FALSE, bPlotFitting=FALSE, method="quadrature");
    con[names(control)] <- control;
    if (con$bPlotFitting) con$bMonitor <- TRUE;
    if (is.factor(t.class) || is.character(t.class)){
        temp <- as.numeric(factor(c(as.character(t.class), as.character(t.class.TEST)),levels=levels(t.class)));
        t.class <- temp[1:length(t.class)];
        t.class.TEST <- temp[(length(t.class)+1):length(temp)];
        rm(temp);
    }
   if (any(theta<=0)) stop("theta params (scale) must be > 0"); 
## ------------------------------------------------------------------------------
    G.LOWER.BOUND.DEFAULT <-  -1e-3; ## default lower bound value
    G.DIFF.DEFAULT        <-  1e100; ## Monitor difference in marginal likelihood
   
    printTrace(paste("Starting at: ", date(), "...... \n"),
        con$sFILE.TRACE, con$InfoLevel, bAppend=FALSE);
    if (nrow(X) == nrow(X.TEST)) {
        b.traintest <- all(X == X.TEST) && all(t.class == t.class.TEST);
    } else  b.traintest <- FALSE;
    Kc <- max(t.class);     
    if(Kc==2){
      t.class[which(t.class==1)]<--1
      t.class[which(t.class==2)]<-1
      t.class.TEST[which(t.class.TEST==1)]<--1
      t.class.TEST[which(t.class.TEST==2)]<-1
      Kc<-1
    }
  ## Identify the number of classes
    N <- nrow(X); Kd <- ncol(X);   ## Get number of samples and dimension of data
    ## randomly initializse M,Y matrix (see paper)
    Y <- matrix(rnorm(N*Kc), nrow=N, ncol=Kc);
    M <- matrix(runif(N*Kc), nrow=N, ncol=Kc);
    ## diagonal matrix of the covariance params for passing to kernel function
    #Theta <- diag(theta);
    ## Set hyper-params for covariance params to one. In this application
    ## I have used a simple exponential distribution over the theta values so
    ## there is only a mean value required psi.
    psi   <- rep(1., length(theta));
    In    <- diag(1., N);          ## N x N dimensional identity matrix
    Ic    <- diag(1., Kc);         ## Kc x Kc dimensional identity matrix
    ## Create the covariance (kernel) matrix and add some small jitter on diagonal
    PHI <- computeKernel(X, X, con$sKernelType, theta) + In * con$nSmallNo;
    ## precompute the inverse matrices required
    invPHI <- chol2inv(chol(PHI + In));
    Ki <- PHI %*% invPHI;
    trace.Ki <- sum(diag(Ki));      
    logDetKi  <- safeLogDet(Ki);   
    logDetPHI <- safeLogDet(PHI);  
    ## Collect all the posterior mean values of the covariance params
    THETA <- matrix(theta, nrow=1, ncol=length(theta))
    ## Collect all the values of the lower-bound
    lowerBound <- G.LOWER.BOUND.DEFAULT;
    ## Monitor difference in marginal likelihood
    scan.diff <- G.DIFF.DEFAULT;
    ## Collect all values of the predictive likelihood
    PL <- NULL;
    ## Collect all values of the percentage predictions incorrect
    testErr <- NULL;    
    ## -------------------------------------------------------------------------
    ## Main loop
    ## -------------------------------------------------------------------------
    its     <- 0;         ## Initiliase iteration number
    bconverged <- FALSE;
    tmean.scan <- NULL;

    if (Kc == 1) {
      tmean   <- tmean.binary;
      genCPP  <- genCPP.binary; 
    }
    else{
    if (con$method == "quadrature") {
      tmean   <- tmean.quad;
      Nsamps  <- con$nNodesQuad;   
      genCPP  <- genCPP.quad;
    } else {
      tmean   <- tmean.classic;
      Nsamps  <- con$nSampsTG;      
      genCPP  <- genCPP.classic;
    }
    }
    while ((its < con$maxIts) && (! bconverged)) {
        its <- its + 1;
        printTrace(paste(its, "> update the columns of the M-matrix ",
            "- equation (8) of the paper"), con$sFILE.TRACE, con$InfoLevel - 1);
        ## - formula (4.6)
        for (k in 1:Kc) M[, k] <- Ki %*% Y[, k];
        printTrace(paste(its, "> update the rows of the Y-matrix",
            "- equation (5) & (6) of the paper "), con$sFILE.TRACE, con$InfoLevel - 1);
        scan.lower.bound <- 0.;
        scan.tm <- NULL;

        for (n in 1:N) {
          #####Esto hay que cambiar!!
            if(Kc>2){
            scan.tm <- tmean(M[n,], t.class[n], Nsamps);
            }else{
            scan.tm <- tmean(M[n,], t.class[n]);
            }
            if (! is.null(scan.tm)) {
                Y[n,] <- scan.tm$tm;
                scan.lower.bound <- scan.lower.bound + safeLog(scan.tm$z);
                rm(scan.tm); scan.tm <- NULL;
            } else stop("tmean error.....");
        }
        if (con$bThetaEstimate) {
            printTrace(paste(its, "> update the posterior mean estimates of the",
                "covariance function parameters and hyper-params"),
                con$sFILE.TRACE, con$InfoLevel - 1);
            ## - formula (4.7)  pag 1796 
            theta <- varphiUpdate(X, M, psi, con$nSampsIS, con$sKernelType);
            #Theta <- diag(theta);
            ## - formula (4.8)  pag 1797
            psi   <- (con$parGammaSigma + 1)/(con$parGammaTau + theta);
            if (con$bMonitor) THETA <- rbind(THETA, theta);
            if (con$bPlotFitting) {
                if (its == 1) par(mfrow=c(2, 2));                
                scov <- matrix(as.numeric(safeLog(THETA)), ncol=ncol(THETA),
                    nrow=nrow(THETA));
                plot(NULL, type="n", xlim=c(1,nrow(scov)), xlab="Iteration",
                     main="Covariance Params Posterior Mean Values",
                     ylim=c(min(scov)-1e-6, max(scov)+1e-6), ylab="log(theta)");
                for (kkk in 1:ncol(scov)) {
                    lines(scov[, kkk],  lty="dotdash", col=kkk);
                }
            }
            PHI <- computeKernel(X, X, con$sKernelType, theta);
            invPHI <- chol2inv(chol(PHI + In));
            Ki <- PHI %*% invPHI;
            trace.Ki  <- sum(diag(Ki));
            logDetKi  <- safeLogDet(Ki);
            logDetPHI <- safeLogDet(PHI);
        }
        printTrace(paste(its, "> compute the lower-bound"), con$sFILE.TRACE,
            con$InfoLevel - 1);
        scan.lower.bound <- scan.lower.bound +
            - 0.5 * Kc * trace.Ki +
            - 0.5 * sum(diag(crossprod(M, invPHI) %*% M)) +
            - 0.5 * Kc * sum(diag(invPHI)) +
            - 0.5 * Kc * logDetPHI  +
            + 0.5 * Kc * logDetKi +
            - 0.5 * Kc * N * safeLog(2*pi) + 0.5*N*Kc + 0.5*N*safeLog(2*pi);
        ## update the development of the bound at every iteration
        lowerBound <- c(lowerBound, scan.lower.bound);
        if (its == 2) lowerBound[1] <- lowerBound[2];
        if (con$bPlotFitting) {
            if (its == 1 && (!con$bThetaEstimate)) par(mfrow=c(1,3));
            plot(lowerBound, type="l", main="Lower Bound",
                lty="dotdash", xlab="Iteration", ylab="Lower bound");
        }
        ## Monitoring convergence
        scan.diff <- abs(100*(scan.lower.bound - lowerBound[its])/lowerBound[its]);
        bconverged <- (scan.diff < con$Thresh);
        if (con$bMonitor || bconverged || its == con$maxIts || con$bPlotFitting) {
            ## --------------------------------------------------------------
            printTrace(paste(its, "> Compute the predictive posteriors on the test set"),
                con$sFILE.TRACE, con$InfoLevel - 1);
            ## Compute the predictive posteriors on the test set and
            ## the associated likelihood and test errors
            Ntest <- nrow(X.TEST);     ## Number of test points
            ## Create test covariance matrices required to obtain predictive
            ## mean and variance values
            if (!b.traintest) {
                PHItest <- computeKernel(X, X.TEST, con$sKernelType, theta);
                PHItestSelf <- computeKernel(X.TEST, X.TEST, con$sKernelType, theta);
                } else PHItest <- PHItestSelf <- PHI;
            ## ---------------------------------------------------------------
            ## Computes the predictive posteriors as defined in Section 4.5 of the paper.
            ## first two equations at page 1798
            Res <- t(crossprod(Y, invPHI)%*%PHItest);
            S <- (diag(PHItestSelf) - diag(crossprod(PHItest, invPHI)%*%PHItest));             
            #I think that Res and S are matrices of dimension testsamples X classes.
           predictive.likelihood <- 0.;
            if(Kc>2){
            Ptest <-  genCPP(Ntest, Kc, Nsamps, Res, S);
            }else{
            Ptest <-  genCPP(Ntest, Kc, Res, S);
            }
            ## Computes the overall predictive likelihood
            if(Kc>2){
           predictive.likelihood <- sum(safeLog( apply(cbind(Ptest,
                t.class.TEST), 1, function(s){s[s[Kc+1]]} )));
            
           if (is.null(PL)) PL <- predictive.likelihood/Ntest
            else             PL <- c(PL, predictive.likelihood/Ntest);
            if (con$bPlotFitting) {
                plot(PL, type="l", main="Predictive Likelihood",
                    lty="dotdash", xlab="Iteration");
            }
            }
            ## Compute the 0-1 error loss.
            fvals <- as.numeric(apply(Ptest, 1, which.max));
            if(Kc==1){
              fvals[which(fvals==1)]<--1
              fvals[which(fvals==2)]<-1             
            }
           
            scanTestErr  <- 100*(sum(fvals != t.class.TEST))/Ntest;
            if (is.null(testErr)) testErr <- scanTestErr
            else testErr <- c(testErr, scanTestErr);
            if (con$bPlotFitting) {
                plot((100-testErr), type="l", lty="dotdash", xlab="Iteration",
                    main="Out-of-Sample Percent Prediction Correct");
            }
            printTrace(paste(its, "> Value of Lower-Bound =", scan.lower.bound,
                ",Prediction Error = ", scanTestErr,
                ", Predictive Likelihood = ", predictive.likelihood/Ntest),
                con$sFILE.TRACE, con$InfoLevel);
            #Acc <- 100 - testErr[its];
            vbmultiprob.obj <- structure( list(
               Ptest=Ptest, X=X, invPHI=invPHI, Y=Y, Kc=Kc, M=M,
               sKernelType=con$sKernelType, THETA=THETA, con=con,
               lowerBound=lowerBound,testErr=testErr, PL=PL),  class="VBMP.obj");
        }
        if (! is.null(con$tmpSave)) save(vbmultiprob.obj, file=con$tmpSave);
    }
    vbmultiprob.obj ;
}

