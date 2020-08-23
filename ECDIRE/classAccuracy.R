#Calculate the accuracy, for each class.
classAccuracy = function(model, testclasscv){
    predictions = predClass(model, numclus)
    lvls = if (numclus == 2) lvls = c(-1,1) else c(1:numclus)
    classes = factor(predictions, levels=lvls)
    testclasscv = factor(testclasscv, levels=lvls)

    #Build confusion matrix
    cmat = table(classes,testclasscv)
    #return accuracy for each class
    return(diag(cmat)/colSums(cmat))
}