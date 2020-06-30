`varphiUpdate` <-
function(X, M, psi, nSamps, sKernelType) {
## This computes the posterior mean of the covariance hyperparameters
## using a simple importance sampler
    V = t(sapply(1:nSamps, function(i) rexp(psi, rate=psi)))
    W = t(apply(V, 1, function(varphi) {
        PHI <- computeKernel(X, X, sKernelType, varphi) +  diag(1., nrow(X));
        return(as.numeric(prod(diag(exp(-0.5*crossprod(M, chol2inv(chol(PHI)))%*% M)))))
    }))
    W <- W/sum(W);
    apply(V, 2, function(v) sum(v*W));
}
