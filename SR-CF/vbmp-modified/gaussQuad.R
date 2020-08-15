`gaussQuad` <-
function(n) {
#
#       Code developed by Gordon Smyth  <smyth@wehi.edu.au>
#
#	Calculate nodes and weights for Guassian quadrature using probability densities.
#	Adapted from Netlib routine gaussq.f
#	Gordon Smyth, Walter and Eliza Hall Institute
#	Corrections for n=1 and n=2 by Spencer Graves, 28 Dec 2005
#	4 Sept 2002. Last modified 4 Jan 2005.
  mu <- 0; sigma <- 1;
	n  <- as.integer(n);
	i  <- 1:n;
	i1 <- 1:(n-1);
	a  <- rep(0,n);
	b  <- sqrt(i1/2);
	A  <- rep(0,n*n);
	A[(n+1)*(i-1)+1]  <- a;
	A[(n+1)*(i1-1)+2] <- b;
	A[(n+1)*i1]       <- b;
	dim(A) <- c(n,n);
	vd <- eigen(A,symmetric=TRUE);
	w <- rev(as.vector( vd$vectors[1,] ))^2;
	x <- rev( vd$values );
  x <- mu + sqrt(2)*sigma*x;
	list(nodes=x,weights=w)
}

