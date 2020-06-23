
suppressMessages(require('R.utils'));
params = commandArgs(asValues=TRUE, 
                     adhoc=TRUE)
data = base::commandArgs(trailingOnly=TRUE)[c(2,3)]
print(params)
print(data[[1]])
print(data[[2]])
