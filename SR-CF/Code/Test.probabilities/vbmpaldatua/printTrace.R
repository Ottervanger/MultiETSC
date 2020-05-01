`printTrace` <-
function(mystr, sFile, InfoLevel, bAppend=TRUE, bStop=FALSE){
    if (InfoLevel > 0) {
        if (length(sFile) > 0) {
            cat(mystr, file=sFile, sep="\n", fill=TRUE, labels=NULL, append=bAppend)
        } else cat(mystr, sep="\n");
        flush.console();
        if (bStop) stop(mystr);
    }
}

