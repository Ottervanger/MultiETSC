`tmean.binary` <-
  function(m, indexMax) {
    ## This function computes the mean of the truncated Gaussian
    ##for the binary case
   #nr<-rep(NA,2)
   #z<-rep(NA,2)
   #tm<-rep(NA,2)
   class<-indexMax
   #for(r in 1:2){
   nr<-class*safeNormPDF(m)
   z<-safeNormCDF(class*m)
   tm<-m+nr/z
   #}
    structure( list( tm=tm, z=z),  class="tmean.obj");
  }
