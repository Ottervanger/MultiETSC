loadData<-function(databasename){

##Access data
load(paste(getwd(),"/databases/",databasename,".RData",sep=""))
data<-database[[1]]
data$class<-database[[2]]
data$class<-factor(data$class)
clus<-unique(database[[2]])
assign("numclus", length(clus), envir = .GlobalEnv)
rm(database)

if(numclus>2){
  levels(data$class)<-c(1:numclus)
}else{
  levels(data$class)<-c(-1,1)
}

## Build training and testing sets set: 

#Train/test
train <- data[data$tt==0,]
test <- data[data$tt==1,]
train$tt<-NULL
test$tt<-NULL
rm(data)

return(list(train,test))
}
