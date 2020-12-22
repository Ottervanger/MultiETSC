`distSqrd` <-
function (X,Y) {
    nx	= nrow(X);
    ny	= nrow(Y);
    abs(rowSums(X^2)%o%rep(1, ny)  +  rep(1, nx)%o%rowSums(Y^2) - 2.*(X %*% t(Y)));
}

