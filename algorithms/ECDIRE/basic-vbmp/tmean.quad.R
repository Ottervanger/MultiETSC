`tmean.quad` <-
function(m, indexMax, Nsamps) {
    ## This function computes the mean of the truncated Gaussian as detailed in
    ## the paper equations (5) & (6).
    UQP <- gaussQuad(Nsamps);
    Kc <- length(m);
    t.class   <- rep(m[indexMax], Kc) - m;
    tr.class  <- t.class;
    t.class   <- t.class[-indexMax];
    ## s ( Nsamps x Kc-1 )
    s  <- matrix(rep(UQP$nodes, Kc-1),byrow=F, nrow=Nsamps, ncol=Kc-1) +
        t(matrix(rep(t.class, Nsamps), byrow=F, ncol=Nsamps));
    if (is.null(dim(s)) || ncol(s) == 1 || nrow(s) == 1) {
        z <- sum(safeNormCDF(as.numeric(s))*UQP$weights);
    } else {
        z <- sum( as.numeric(apply(safeNormCDF(s), 1, prod))*UQP$weights);
    }
    if (Kc > 2) {
        tm <- rep(NA, Kc);
        for (r in 1:Kc) {
            ## sr ( Nsamps x Kc )
            sr <- matrix(rep(UQP$nodes, Kc), byrow=F, nrow=Nsamps) +
                t(matrix(rep(tr.class,Nsamps), byrow=F, ncol=Nsamps));
            sr <- sr[, -c(r, indexMax)];
            if (is.null(dim(sr)) || ncol(sr) == 1 || nrow(sr) == 1) {
                snr <- as.numeric(safeNormCDF(sr));
            } else {
                snr <- as.numeric(apply(safeNormCDF(sr), 1, prod));
            }
            nr <- sum(UQP$weights*safeNormPDF(UQP$nodes + m[indexMax] - m[r]) * snr );
            if (r == indexMax) tm[r] <- 0.
            else               tm[r] <- m[r] - nr/z;
        }
        tm[indexMax] <- sum(m) - sum(tm);
    } else {
        stop('Multinomial only code !!!');
    }
    structure( list( tm=tm, z=z),	class="tmean.obj");
}

