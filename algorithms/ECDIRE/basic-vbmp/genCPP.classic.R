`genCPP.classic` <-
function(Ntest, Kc, Nsamps, Res, S) {
   Ptest <- matrix(1., nrow=Ntest, ncol=Kc);
   u     <- rnorm(Nsamps);
   for (n in 1:Ntest) {
      for (i in 1:Kc) {
         pp <- rep(1., Nsamps);
         for (j in ((1:Kc)[-i])) {
            pp <- pp * safeNormCDF(u + (Res[n, i] - Res[n, j])/(sqrt(1.+S[n])));
         }
         Ptest[n, i] <- mean(pp);
      }
   }
   t(apply(Ptest, 1, function(x) {x/sum(x)})); ## JUST IN CASE
}

