`rexponential` <-
function(lambda) {
    #-log(runif(length(lambda)))/lambda
    rexp(rep(1,length(lambda)), rate=lambda);
}

