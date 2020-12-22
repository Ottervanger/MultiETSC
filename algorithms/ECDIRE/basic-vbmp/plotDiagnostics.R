`plotDiagnostics` <-
function(obj) {
   if (nrow(obj$THETA) > 1)  {
      par(mfrow=c(2, 2));
      ## plot covariance parameters evolution
      scov <- matrix(as.numeric(safeLog(obj$THETA)), ncol=ncol(obj$THETA), 
         nrow=nrow(obj$THETA));
      plot(NULL, type="n", xlim=c(1,nrow(scov)), xlab="Iteration",
         main="Covariance Parameters",
         ylim=c(min(scov)-1e-6, max(scov)+1e-6), ylab="log(theta)");
      for (kkk in 1:ncol(scov)) {
          lines(scov[, kkk],  lty="dotdash", col=kkk);
      }   
   } else par(mfrow=c(1, 3));
   ## plot lower bound evolution
   plot(obj$lowerBound, type="l", main="Lower Bound", lty="dotdash", 
      xlab="Iteration", ylab="Lower bound");
   ## plot PL evolution
   plot(obj$PL, type="l", main="Predictive Likelihood", lty="dotdash", 
      xlab="Iteration", ylab="PL");
   ## plot test error
   plot((100-obj$testErr), type="l", lty="dotdash", xlab="Iteration", 
      ylab="Accuracy %", main="Out-of-Sample Prediction Correct");
   par(mfrow=c(1,1))
}

