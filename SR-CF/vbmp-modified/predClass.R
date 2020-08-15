`predClass` <-
function(obj,Kc) {
    fvals<-as.numeric(apply(obj$Ptest, 1, which.max));
    if(Kc==2){
      fvals[which(fvals==1)]<--1
      fvals[which(fvals==2)]<-1 
    }
  return(fvals)
}

