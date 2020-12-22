`safeLog` <-
function(x) {
    G.THRESH.LOG      <-  1e-200; ## threshold used to avoid numerical problem
    x[x < G.THRESH.LOG]    <- G.THRESH.LOG;
    x[x > 1./G.THRESH.LOG] <- 1./G.THRESH.LOG;
    log(x);
}

