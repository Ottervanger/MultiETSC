computeKernel = function (X1, X2, kName, kParams) {
    X1 <- t(t(X1)/sqrt(kParams))
    X2 <- t(t(X2)/sqrt(kParams))
    switch(kName,
        'iprod'   = tcrossprod(X1, X2),
        'gauss'   = exp(-distSqrd(X1, X2)),
        'cauchy'  = 1./(1. + distSqrd(X1, X2)),
        'laplace' = exp(-sqrt(abs(distSqrd(X1, X2)))),
        stop(paste('kernel',kName,'not implemented'))
    )
}