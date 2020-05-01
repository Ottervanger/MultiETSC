`varphiUpdate` <-
function(X, M, psi, nSamps, sKernelType) {
## This computes the posterior mean of the covariance hyperparameters
## using a simple importance sampler
    V <- NULL;
    W <- NULL;
    for (i in 1:nSamps) {
        varphi <- rexponential(psi);
        #Varphi <- diag(varphi);
        PHI <- computeKernel(X, X, sKernelType, varphi) +  diag(1., nrow(X));
        invPHI <- chol2inv(chol(PHI)); # solve(PHI);
        ws <- as.numeric(prod(diag(exp(-0.5*crossprod(M, invPHI)%*% M))));
        if (is.null(V)) V <- matrix(varphi, nrow=1, ncol=length(varphi))
        else  V <- rbind(V, varphi);
        if (is.null(W)) W <- ws
        else  W <- c(W, ws);   
    }
    W <- W/sum(W);
    colSums(V * matrix(rep(W, ncol(V)), byrow=FALSE, nrow=length(W), ncol=ncol(V)));
}

