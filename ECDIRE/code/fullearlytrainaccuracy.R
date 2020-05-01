#Calculate the accuracy, for each class.

obtainaccuracy<-function(model){
    
    #True and predicted class values saves as factors
    if(numclus>2){
    classes<-factor(predClass(model,numclus),levels=c(1:numclus))
    testclasscv<-factor(testclasscv,levels=c(1:numclus))
    }else{
    classes<-factor(predClass(model,numclus),levels=c(-1,1))
    testclasscv<-factor(testclasscv,levels=c(-1,1))
    }
    #Build confusion matrix
    confusionmatrix<-table(classes,testclasscv)
    #Calculate accuracy for each class
    accuracy<-as.numeric(diag(confusionmatrix))
    accuracy<-accuracy/colSums(confusionmatrix)
return(accuracy)
}