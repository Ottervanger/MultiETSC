`safeNormPDF` <-
function(x) {
    G.THRESH.NORM.PDF <-  35;     ## threshold used to avoid numerical problem
    x[x < -G.THRESH.NORM.PDF] <- -G.THRESH.NORM.PDF;
    x[x >  G.THRESH.NORM.PDF] <-  G.THRESH.NORM.PDF;
    dnorm(x);
}

