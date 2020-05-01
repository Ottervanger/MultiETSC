# Generate a list of partitions for a cross-validation that can
# be used for all cross-validation methods. 
# For parameters, see crossValidation
generateCVRuns <- function(labels, ntimes = 10, nfold = 10, 
                           leaveOneOut=FALSE, stratified = FALSE)
{
  if(leaveOneOut)
    ntimes <- 1
  
  numSamples <- length(labels)
  res <- lapply(1:ntimes,function(run)
    # for each run
  {
    # calculate folds
    if (leaveOneOut)
    {
      indices <- as.list(1:numSamples)
    }
    else
    {
      if(stratified)
      {
        classes <- unique(labels)
        sing.perm <- lapply(classes, function(cl){
          index <- which(labels == cl)
          sample(index, length(index))
        })
        permut <- unlist(sing.perm)
        indices <- lapply(1:nfold,function(i){c()})
        for(i in 1:numSamples)
        {
          k = i%%nfold
          if(k==0)
            k = nfold
          
          indices[[k]] <- c(indices[[k]], permut[i])
        }
      }
      else
      {
        # permute the indices of the samples
        permut <- sample(1:numSamples, numSamples,replace=FALSE)
        indices <- lapply(1:nfold, function(i)
        {
          # split the samples in nfold groups
          permut[seq(i, numSamples, nfold)]
        })
      }
    }
    names(indices) <- paste("Fold ",1:nfold)
    return(indices)
  })
  names(res) <- paste("Run ", 1:ntimes)
  return(res)
}