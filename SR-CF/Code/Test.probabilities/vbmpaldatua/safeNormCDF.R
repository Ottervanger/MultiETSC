`safeNormCDF` <-
function(x) {
    G.THRESH.NORM.CDF <-  10;     ## threshold used to avoid numerical problem
    x[x < -G.THRESH.NORM.CDF] <- -G.THRESH.NORM.CDF;
    pnorm(x);
}

