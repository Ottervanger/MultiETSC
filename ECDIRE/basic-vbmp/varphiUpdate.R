`varphiUpdate` <-
function(X, M, psi, nSamps, sKernelType) {
## This computes the posterior mean of the covariance hyperparameters
## using a simple importance sampler
    V = t(replicate(nSamps, rexp(psi, rate=psi)))
    w = apply(V, 1, function(varphi) {
        PHI <- computeKernel(X, X, sKernelType, varphi) +  diag(1., nrow(X));
        exp(-0.5*sum(diag(crossprod(M, chol2inv(chol(PHI)))%*% M)))
    })
    w <- w/sum(w);
    colSums(V*w)
}
