
#This function checks the second level reliability.

#model=GP model that we have trained

checkrel2<-function(model,targetclasses,classes,reliability2,root,numclus){
    
  #Initialization
  classified<-c()#indices of classified instances
  class<-c()#Their class
  
  ####################MULTICLASS DATABASES###################
  if(numclus!=2){
    
    #Only target classes are considered
    for(j in targetclasses){ 
      
      classifiedaux<-which(classes==j)
      if(length(classifiedaux)==0){next}
      
      #extract posterior probabilities
      probs<-matrix(model$Ptest[classifiedaux,],ncol=numclus,byrow=FALSE)
      
      #select minimum difference (closest class)
      if(dim(probs)[1]==1){
        probs<-min((probs[,j]-probs)[,-j]) 
      }else{
        probs<-apply((probs[,j]-probs)[,-j],1,min)
      }
      #check if threshold is met and discard elements that don't
      nocheck<-which(probs<reliability2[,j])
      if(length(nocheck)!=0){
        classifiedaux<-classifiedaux[-nocheck]}
      classified<-c(classified,classifiedaux)
      
      #Define class of classified instances
      classaux<-rep(j,length(classifiedaux))
      class<-c(class,classaux)
    }
    
 #####################BINARY DATABASES############################   
}else{
    #ONLY ONE CLASS IS CONSIDERED
    if(root==FALSE){
    classifiedaux<-which(classes==c(-1,1)[targetclasses])
    #posterior probabilities are extracted
    probs<-matrix(model$Ptest[classifiedaux,],ncol=numclus,byrow=FALSE)
    probs<-(probs[,targetclasses]-probs)[,-targetclasses]
    #check when threhold is met and discard instances that do not.
    nocheck<-which(probs<reliability2[,targetclasses])
    if(length(nocheck)!=0){
    classifiedaux<-classifiedaux[-nocheck]}
    classified<-classifiedaux
    #Define classes for classified instances
    classaux<-rep(c(-1,1)[targetclasses],length(classifiedaux))
    class<-c(class,classaux)
  }else{
    #BOTH CLASSES TAKEN INTO ACCOUNT
    for(j in c(-1,1)){ 
      if(j==-1){jj<-1
      }else{jj<-2}
      classifiedaux<-which(classes==j)
      if(length(classifiedaux)==0){next}
      #Select posterior probabilities
      probs<-matrix(model$Ptest[classifiedaux,],ncol=numclus,byrow=FALSE)
      probs<-(probs[,jj]-probs)[,-jj]
      #Check wether threshold is met and discard instances that don't.
      nocheck<-which(probs<reliability2[,jj])
      if(length(nocheck)!=0){
        classifiedaux<-classifiedaux[-nocheck]}
      classified<-c(classified,classifiedaux)
      #Define classes for classified instances
      classaux<-rep(j,length(classifiedaux))
      class<-c(class,classaux)
    }
  }
}
  
  return(list(classified,class))
}



