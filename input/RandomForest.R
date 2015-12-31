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
library(randomForest)
library(factoextra)


setwd(".")

traps <- unique(merged_agg_test$TrapFactor)
species <- unique(merged_agg_test$SpeciesFactor) 
#Random Forests
random_fst_data <- fread("merged_data.csv", data.table = T, stringsAsFactors=T)

predictand <- cbind(as.factor(ifelse(random_fst_data$V2 > 0, 1, 0)))
predictors <- cbind(
  factor(random_fst_data$TrapFactor, levels=traps),
  factor(random_fst_data$SpeciesFactor, levels=species),
  factor(random_fst_data$ReadingDate),
  random_fst_data$weekNumber, 
  random_fst_data$Tavg,
  random_fst_data$Tmax, 
  random_fst_data$Tmin,
  random_fst_data$PrecipTotal, 
  random_fst_data$ThreedaymedianMinTmp,
  random_fst_data$ThreedaymedianTmp,
  random_fst_data$AvgSpeed,
  random_fst_data$StnPressure
                    )

cor(predictors, predictand)


class_rnd_fst_fit <- randomForest(x=predictors, y=factor(predictand), ntree=100)
plot(class_rnd_fst_fit)
print(class_rnd_fst_fit)
importance(class_rnd_fst_fit)

random_fst_data$predict <- predict(class_rnd_fst_fit, predictors, type="prob")
res <- subset(random_fst_data, WnvPresentChar==T)
predictors_df <- data.frame(predictors)
colnames(predictors_df) <-  c("TrapFactor", "SpeciesFactor", "ReadingDate","weekNumber",  "Tavg", 
                               "PrecipTotal", "ThreedaymedianMinTmp", 
                              "ThreedaymedianTmp", "AvgSpeed", "StnPressure")
melted_predictors <- melt(predictors_df)


temperature_plot <- ggplot(subset(melted_predictors, variable == "Tavg" | variable == "ThreedaymedianMinTmp" | variable == "ThreedaymedianTmp" ), aes(x=variable, y=value))+geom_boxplot(aes(fill=variable))+theme(legend.position="none")+ylab("Degrees F")+xlab("")
print(temperature_plot)
other_plot <- ggplot(subset(melted_predictors, variable == "PrecipTotal" ), aes(x=variable, y=value))+geom_boxplot(aes(fill=variable))+theme(legend.position="none")+ylab("Inches")+xlab("")
print(other_plot)

speed_plot <- ggplot(subset(melted_predictors, variable == "AvgSpeed" ), aes(x=variable, y=value))+geom_boxplot(aes(fill=variable))+theme(legend.position="none")+ylab("Miles/Hr")+xlab("")
print(speed_plot)

pressure_plot <- ggplot(subset(melted_predictors, variable == "StnPressure" ), aes(x=variable, y=value))+geom_boxplot(aes(fill=variable))+theme(legend.position="none")+ylab("PSI")+xlab("")
print(pressure_plot)


#| variable == "AvgSpeed" | variable == "StnPressure" 


predictor_plot <- ggplot(subset(melted_predictors, variable != "dist_to_recent_spray" & variable != "SpeciesFactor" & variable != "TrapFactor" & variable != "weekNumber"), aes(x=variable, y=value))+geom_boxplot(aes(fill=variable))+theme(legend.position="none")
print(predictor_plot)




pca <- princomp( ~ dist_to_recent_spray+Tavg+StnPressure+PrecipTotal+Tmax+Tmin+SeaLevel+AvgSpeed, data=predictors_df, cor=T, scores=T)
#Plot PCA contributions
viz_pca_var(pca, col.var="contrib") +
  scale_color_gradient2(low="white", mid="blue", 
                        high="red", midpoint=50) + theme_minimal()
var <- get_pca_var(pca)
print(var$coord)
