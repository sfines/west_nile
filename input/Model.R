library(data.table)
library(dplyr)
library(ggplot2)
library(magrittr)
library(maps)
library(zoo)
library(stats)
library(reshape2)
library(geosphere)

library(ROCR)
library(grid)

library(cvTools)
library(aod)




predictand <- cbind(merged_agg_train$V2)
predictors <- cbind(merged_agg_train$weekNumber, merged_agg_train$dist_to_recent_spray, 
                    merged_agg_train$Tavg, merged_agg_train$StnPressure, merged_agg_train$PrecipTotal, 
                    factor(merged_agg_train$TrapFactor, levels=traps ), merged_agg_train$Tmax, merged_agg_train$Tmin,
                    merged_agg_train$SeaLevel, factor(merged_agg_train$SpeciesFactor, levels=species), merged_agg_train$AvgSpeed)

model <- glm(factor(V2) ~ weekNumber +Tavg+StnPressure+PrecipTotal+AvgSpeed+dist_to_recent_spray, 
             data=merged_agg_train, 
             family=binomial(link="logit"))

pcs <- princomp(predictors)
plot(pcs)

k <- 10 # number of validation folds

folds <- cvFolds(NROW(merged_agg_train), K=k)
merged_agg_train$holdout <- rep(0, nrow(merged_agg_train))

for( i in 1:k){
  tf <- merged_agg_train[folds$subsets[folds$which != i], ] #training set
  tf$SpeciesFactor <- factor(tf$SpeciesFactor, levels=species)
  tf$TrapFactor <- factor(tf$TrapFactor, levels=traps )
  validation <- merged_agg_train[folds$subsets[folds$which== i ], ] #validation set
  validation$SpeciesFactor <- factor(validation$SpeciesFactor, levels=species)
  validation$TrapFactor <- factor(validation$TrapFactor, levels=traps)

  model <- glm( 
    WnvPresentChar ~ TrapFactor +SpeciesFactor+ weekNumber +Tavg+StnPressure+PrecipTotal+AvgSpeed+dist_to_recent_spray, 
    data=tf, 
    family=binomial(link="logit"))
  
  newpred <- predict(model, newdata=validation)
  
  merged_agg_train[folds$subsets[folds$which == i], ]$holdoutpred <- newpred
}


ci_model_data <- cbind(merged_agg_train, predict(model, newdata=merged_agg_train, type="link", se=T))
ci_model_data <- within(ci_model_data, {
  predictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

predObj <- prediction(merged_agg_train, merged_agg_train$WnvPresentChar)
precObj <- performance(predObj, measure="prec")
recObj <- performance(predObj, measure="rec")

precision <- (precObj@y.values)[[1]]
prec.x <- (precObj@x.values)[[1]]
recall <- (recObj@y.values)[[1]]

rocFrame <- data.frame(threshold=prec.x, precision=precision, recall=recall)

nplot <- function(plist){
  n <- length(plist)
  grid.newpage()
  pushViewport(viewport(layout=grid.layout(n,1)))
  vplayout=function(x,y){viewport(layout.pos.row=x, layout.pos.col=y)}
  for(i in 1:n){
    print(plist[[i]], vp=vplayout(i,1))
  }
}

pnull <- mean(as.numeric(merged_agg_train$WnvPresentChar))
p1 <- ggplot(na.omit(rocFrame), aes(x=threshold))+geom_line(aes(y=precision/pnull))

p2 <- ggplot(rocFrame, aes(x=threshold))+geom_line(aes(y=recall))

nplot(list(p1, p2))
print(p1)
print(p2)

