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
#library(grid)
#library(factoextra)

library(party)
library(partykit)

setwd(".")

traps <- unique(merged_agg_test$TrapFactor)
species <- unique(merged_agg_test$SpeciesFactor) 
#Random Forests
random_fst_data <- fread("merged_data.csv", data.table = T, stringsAsFactors=T)
random_fst_data$TrapFactor <- factor(random_fst_data$TrapFactor, levels=traps)
random_fst_data$SpeciesFactor <- factor(random_fst_data$SpeciesFactor, levels=species)
random_fst_data$ReadingDate <- factor(random_fst_data$ReadingDate)

set.seed(13147)

data = ctree(V2 ~ TrapFactor+SpeciesFactor+weekNumber+Tavg+Tmax+Tmin+PrecipTotal+ThreedaymedianMinTmp+ThreedaymedianTmp+AvgSpeed+StnPressure , data = random_fst_data, control = ctree_control( mtry=4))

km_curve <- treeresponse(data.rf, newdata=random_fst_data[1:2, ], OOB = T)
plot(varimp(data.rf))
plot(data.rf)
print(data.rf)