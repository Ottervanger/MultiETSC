`genCPP.quad` <-
function(Ntest, Kc, Nsamps, Res, S) {
   UQP <- gaussQuad(Nsamps);
   Res <- Res / sqrt(1. + S);
   Ptest <- t(apply(Res,1, function(Res.n) {
      Ptest.n <- rep(0,Kc);
      for (i in 1:Kc) {
         Res.i <- UQP$nodes + Res.n[i];
         pp    <- UQP$weights * apply(sapply(Res.n[-i],
             function(vb) safeNormCDF(Res.i - vb)), 1, prod);
         Ptest.n[i] <- sum(pp);
      }
      Ptest.n;}
   ));
   t(apply(Ptest, 1, function(x) {x/sum(x)})); ## JUST IN CASE
}

