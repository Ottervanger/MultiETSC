`genCPP.binary` <-
  function(Ntest, Kc, Res, S) {
    Ptest <- matrix(1., nrow=Ntest, ncol=2);
    for (n in 1:Ntest) {
      pp<-safeNormCDF(Res[n]/(sqrt(1.+S[n])));
      Ptest[n, 2] <- pp;
      Ptest[n, 1] <-1-Ptest[n, 2]
    }
    #print(Ptest)
    Ptest
  }
