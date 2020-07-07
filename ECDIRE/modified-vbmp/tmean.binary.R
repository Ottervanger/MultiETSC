`tmean.binary` <-
  function(m, indexMax) {
    ## This function computes the mean of the truncated Gaussian
    ##for the binary case
   class<-indexMax
   nr<-class*safeNormPDF(m)
   z<-safeNormCDF(class*m)
   tm<-m+nr/z
    structure( list( tm=tm, z=z),  class="tmean.obj");
  }
