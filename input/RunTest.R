##
## This is the actual test predictions to post to Kaggle
##

library(data.table)
library(dplyr)
library(ggplot2)
library(magrittr)
library(maps)
library(zoo)
library(stats)
library(reshape2)
library(geosphere)

options(stringsAsFactors=T)

chicago_midway_airpt <- c(41.7859722, -87.7524167)
chicago_ohare_airpt <- c(41.9793333, -87.9073889)

MAX_SPRAY_DAYS <- 14

setwd("~/OneNote Notebooks/MDS549 - Data Mining Project/west_nile/input/")


nearbyAirport<- Vectorize(function(lat, long){
  point = c(lat,long)
  dist_ohare = distMeeus(point, chicago_ohare_airpt)
  dist_mdwy = distMeeus(point, chicago_midway_airpt)
  
  if( dist_mdwy < dist_ohare){
    2
  } else {
    1
  }
})

 remove(merged_agg_test)
spray_series <- fread("spray.csv", data.table = T, stringsAsFactors=T)
weather_series <- fread("weather-both-trimmed.csv", data.table = T, stringsAsFactors=T)

test <- fread("test.csv", data.table = T, stringsAsFactors=T)
test$ReadingDate <- as.Date.factor(test$Date, "%Y-%m-%d")
test$year <- as.numeric(format(as.Date(test$Date), "%Y"))
test$weekNumber <- as.numeric(format(as.Date(test$Date), "%W"))
test$Station <- nearbyAirport(test$Latitude, test$Longitude)
weather_series$ReadingDate <- as.Date.factor(weather_series$Date, "%Y-%m-%d")
weather_series$SevendaymedianTmp <- rollmedian(weather_series$Tavg, 7 )
weather_series$ThreedaymedianTmp <- rollmedian(weather_series$Tavg, 3 )
weather_series$SevendaymedianMinTmp <- rollmedian(weather_series$Tmin, 7 )
weather_series$ThreedaymedianMinTmp <- rollmedian(weather_series$Tmin, 3 )

# The data dictionary says that the records are broken up when there are More than 50 in a trap, so 
# we're re-aggregating them. Any number > 0 in the WnvPresent column means that there were mosquitos 
# that tested positive, not how many there were.
#agg_test <- test[, .(sum(0), sum(0)), by=.(TrapFactor, ReadingDate, SpeciesFactor, Latitude, Longitude, weekNumber, Station)][order(TrapFactor, ReadingDate, SpeciesFactor, Latitude, Longitude, weekNumber, Station)]

#agg_test$WnvPresentChar <- ifelse(agg_test$V2 > 0, T, F)


#This checks if a spraying event occured recently

recentSpray<- function (chk_Date){
  trap_date <- as.Date(chk_Date, "%Y-%m-%d")
  spray_min_date <- trap_date - MAX_SPRAY_DAYS
  spray_max_date <- trap_date
  sprays_nearby_in_time <- subset( spray_series, as.Date(spray_series$Date, "%Y-%m-%d") >= spray_min_date & as.Date(spray_series$Date, "%Y-%m-%d") <= spray_max_date)
  
  sprays_nearby_in_time
}

#This finds all sprays that happened close to the point
nearbySpray<- function(lat, long, spray_data){
  spray_loc_matrix <- data.matrix(cbind(spray_data$Latitude,spray_data$Longitude))
  distance = distMeeus(c(lat, long), spray_loc_matrix)
  spray_data$distance <- distance #one half mile is (roughly) 805m
}




impactingSpray <- Vectorize(function( date, lat, long){
  
  spray <- recentSpray(date)
  if( nrow(spray) > 0){
    spray$distance <- nearbySpray(lat, long, spray)
    min(spray$distance)
  } else { 
    100000
  }
})

#impact_spray <- Vectorize(impactingSpray)

merged_agg_test<- inner_join(test, weather_series,by=c("Station", "ReadingDate"))

merged_agg_test$TrapFactor <- factor(merged_agg_test$TrapFactor)
merged_agg_test$SpeciesFactor <- factor(merged_agg_test$SpeciesFactor)
merged_agg_test$ReadingDate <- factor(merged_agg_test$ReadingDate)
write.csv(merged_agg_test, "merged_test.csv")


predictors <- data.frame(cbind(
  factor(merged_agg_test$TrapFactor, levels=traps),
  factor(merged_agg_test$SpeciesFactor, levels=species),
  factor(merged_agg_test$ReadingDate),
  merged_agg_test$weekNumber, 
  merged_agg_test$Tavg,
  merged_agg_test$Tmax,
  merged_agg_test$Tmin,
  merged_agg_test$PrecipTotal, 
  merged_agg_test$ThreedaymedianMinTmp,
  merged_agg_test$ThreedaymedianTmp,
  merged_agg_test$AvgSpeed,
  merged_agg_test$StnPressure
  ))

names(predictors) <- c("TrapFactor", "SpeciesFactor", "ReadingDate", "weekNumber", "Tavg",
                       "Tmax", "Tmin", "PrecipTotal","ThreedaymedianMinTmp", "ThreedaymedianTmp", 
                       "AvgSpeed", "StnPressure")

library(party)

merged_agg_test$predict <- predict(data.rf, newdata=predictors, OOB=T)
merged_agg_test$pred<- 1-merged_agg_test$predict

## use LogisticRegression
submRes <- data.frame(cbind(merged_agg_test$pred))
colnames(submRes) <- c("WnvPresent")
write.csv(x = submRes, file="submission-07122015-7.csv")
