loadData<-function(datapath){

    ##Access data
    d = read.table(datapath)
    data = list()
    data$ts = d[,-1]
    data$class = factor(d[,1])

    # TODO: numclus is here globally defined and used throughout 
    # run :grep numclus code/.R to show 
    assign("numclus", nlevels(data$class), envir = .GlobalEnv)

    # GO I dont see the problem with class labels c(1,2) why add this elaborate code?
    if(numclus>2){
        levels(data$class)<-c(1:numclus)
    }else{
        levels(data$class)<-c(-1,1)
    }

    return(data)
}
