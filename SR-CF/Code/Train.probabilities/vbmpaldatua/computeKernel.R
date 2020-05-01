`computeKernel` <-
function(X1, X2, kName, kParams){
    ##
    ##	X1	     N1 x d data matrix
    ##	X2	     N2 x d data matrix
    ##  kName    string name of kernel function
    ##           Prefix with '+' to add bias (e.g. '+gauss')
    ##		       Prefix with '*' to add bias and augmenting with X2 covariate (e.g. '*gauss')
    ##  kParams  vector parameters of kernel functions
    ##
    ##	returns N1 x N2 (or N2+1 if bias, or N2+ncol(X2)+1 for X2 augmentation ) design matrix.
    ##			The first column comprises 1's if a bias is used
    ##      The subsequent ncol(X2) columns comprises X2 covariate if X2 augmentation is used
    N1 <- nrow(X1); d <- ncol(X1);
    N2 <- nrow(X2);
    if (length(kParams) == 1) kParams <- rep(kParams,d);
    varParams <- NULL;
    if (substr(kName, 1, 1) == '+') {
        b <- matrix(1., nrow=N1, ncol=1);
        kName	<- substr(kName, 2, nchar(kName));
    } else {
        if (substr(kName, 1, 1) == '*') {
            b <- as.matrix(cbind(rep(1.,N1), X1));
            kName <- substr(kName, 2, nchar(kName));
            if (length(kParams) == d)  {
                varParams <- c(1., rep(1,d));
            } else {
                # rescaling X2 compontent differently from that used for kernel
                varParams <- c(1., kParams[1:d]);
                kParams <- kParams[(d+1):(2*d)];
            }
        } else b <- NULL;
    }
    ##    poly' --> Polynomial   e.g. "poly5"
    if (nchar(kName)>=4 && substr(kName, 1, 4) == 'poly') {
        p <- as.numeric(substr(kName, 5, nchar(kName)));
        kName	= 'poly';
    }     
    ##    'hpoly' --> Homogeneous Polynomial
    if (nchar(kName)>=5 && substr(kName, 1, 5) == 'hpoly') {
        p <- as.numeric(substr(kName, 6, nchar(kName)));
        kName	<- 'hpoly';
    }
    X1 <- t(as.matrix(apply(X1, 1, function(x) x/sqrt(kParams))));
    X2 <- t(as.matrix(apply(X2, 1, function(x) x/sqrt(kParams))));
    if (kName =='gauss') {                 ##	Gaussian
        PHI <- exp(-distSqrd(X1, X2));
    } else if (kName == 'iprod') {         ##	Inner product
        PHI <- tcrossprod(X1, X2); 
    } else if (kName == 'tps') {           ##	'Thin-plate' spline
        r2  <- distSqrd(X1, X2);
        PHI <- 0.5 * r2 *log(sqrt(r2)); 
    } else if (kName == 'cauchy') {        ##	Cauchy (heavy tailed) in distance
        r2  <- distSqrd(X1, X2);
        PHI <- 1./(1. + r2);
    } else if (kName == 'cubic') {         ##	Cube of distance
        r2 <- distSqrd(X1, X2);
        PHI <- r2 * sqrt(r2);
    } else if (kName == 'r') {             ##	Distance
        PHI <- sqrt(distSqrd(X1, X2));
    } else if (kName == 'neighbour') {     ## Neighbourhood indicator
        r2  <- distSqrd(X1, X2);
        PHI <- ifelse(r2 < 1, 1, 0);
    } else if (kName == 'laplace'){        ##	Laplacian
        r2  <- distSqrd(X1, X2);
        PHI <- exp(-sqrt(r2));
    }  else if (kName == 'poly') {         ##  polinomial (X + 1)^p;
        PHI <- (X1%*%t(X2) + 1)^p;
    }  else if (kName == 'hpoly') {        ##  homogeneous polynomial  X^p;
        PHI <- (X1%*%t(X2))^p;              
    } else if (kName == 'lsp') {           ##  'linear' spline kernel
        PHI <- 1.;
        for (i in 1:d) {
            XX  <- X1[,i] %o% X2[, i];
            Xx1 <- X1[,i] %o% rep(1, N2);
            Xx2 <- rep(1,N1) %o% X2[, i];
            minXX <- Xx2; 
            minXX[Xx1 - Xx2 > 0] <- Xx1[Xx1 - Xx2 > 0];
            PHI <- PHI * (1. + XX + XX*minXX - ((Xx1 + Xx2)/2.)*(minXX^2) + (minXX^3)/3.);
        }
    }
    if (any(b)) {
        if (length(varParams) > 0) b <- t(as.matrix(apply(b, 1, function(x) x/varParams)));
        PHI <- as.matrix(cbind(b, PHI));
    }
    PHI;
}

