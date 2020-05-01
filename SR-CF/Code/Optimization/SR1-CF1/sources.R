######################FUNCTIONS#####################################

#Simple linear rule. Only one for all classes.
s<-function(probabilities, sigma){
  probabilities[1:(length(probabilities)-1)]<-sort(probabilities[1:length(probabilities)-1], decreasing=TRUE)
  rule<-sigma[1]*probabilities[1]+sigma[2]*(probabilities[1]-probabilities[2])+sigma[3]*probabilities[length(probabilities)]
  prediction<-(rule>0)
  return(prediction)
}


#Cost of earlyness
cost.earlyness<-function(t){
  c<-1
  ce<-c*t
  return(ce)
}

#Cost of accuracy
cost.accuracy<-function(predicted.class, true.class, regret){
  
  if(predicted.class==true.class){
    ca<-0
  }else{
    ca<-regret
  }
  return(ca)
}


#Complete cost function
cost_uniclass<-function(sigma, probabilities, alpha, regret,cl){
  
  #We have a linear function s, which takes the probabilities 
  #issued by the f classifiers and provides an answer: halt or not halt
  #Halt function
  
  #Initialize costs
  ce<-0
  ca<-0
  
  for(i in c(1:length(probabilities))){
    
    #Probabilities is a table, where each row corresponds to a time and we 
    #have the following information p1,p2,...,pk,t
    probs<-probabilities[[i]]
    trueclasses<-unique(probs$class)
    probs$class<-NULL
    probs<-probs[,ordervariables]
    
    
    #For all the probabilities obtained for a series (at each timestamp, one set), we extract the first one in
    #which s HALTS
    predictions<-apply(cbind(probs,c(1:20)/20),1,s,sigma)
    t<-which(predictions==1)[1]
    if(is.na(t)){t<-20}
    t<-t/20
    
    #Add the cost of this series to the total expected cost
    ce<-ce+cost.earlyness(t)
    predictedclass<-as.numeric(which.max(probs[t*20,]))
    predictedclass<-cl[predictedclass]
    ca<-ca+cost.accuracy(predictedclass,trueclasses[1], regret)
  }
  
  c<-alpha*ca+(1-alpha)*ce
  return(c)
}

