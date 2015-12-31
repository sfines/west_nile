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


spray_series <- fread("spray.csv", data.table = T, stringsAsFactors=T)
weather_series <- fread("weather-both-trimmed.csv", data.table = T, stringsAsFactors=T)

train <- fread("train.csv", data.table = T, stringsAsFactors=T)
train$TrapFactor <- factor(train$Trap)
train$SpeciesFactor <- factor(train$Species)
train$ReadingDate <- as.Date.factor(train$Date, "%Y-%m-%d")
train$year <- as.numeric(format(as.Date(train$Date), "%Y"))
train$weekNumber <- as.numeric(format(as.Date(train$Date), "%W"))
train$Station <- nearbyAirport(train$Latitude, train$Longitude)
weather_series$ReadingDate <- as.Date.factor(weather_series$Date, "%Y-%m-%d")
weather_series$SevendaymedianTmp <- rollmedian(weather_series$Tavg, 7 )
weather_series$ThreedaymedianTmp <- rollmedian(weather_series$Tavg, 3 )
weather_series$SevendaymedianMinTmp <- rollmedian(weather_series$Tmin, 7 )
weather_series$ThreedaymedianMinTmp <- rollmedian(weather_series$Tmin, 3 )
# The data dictionary says that the records are broken up when there are More than 50 in a trap, so 
# we're re-aggregating them. Any number > 0 in the WnvPresent column means that there were mosquitos 
# that tested positive, not how many there were.
agg_train <- train[, .(sum(NumMosquitos), sum(WnvPresent)), by=.(TrapFactor, ReadingDate, SpeciesFactor, Latitude, Longitude, weekNumber, Station)][order(TrapFactor, ReadingDate, SpeciesFactor, Latitude, Longitude, weekNumber, Station)]

agg_train$WnvPresentChar <- ifelse(agg_train$V2 > 0, T, F)

betweenDates <- function( targetDate, startDate, endDate){
  targetDate >= startDate && targetDate <= endDate
}

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

#agg_train$dist_to_recent_spray <- impactingSpray(agg_train$ReadingDate, agg_train$Latitude, agg_train$Longitude)
merged_agg_train <- inner_join(agg_train, weather_series,by=c("Station", "ReadingDate"))
write.csv(merged_agg_train, "merged_data.csv")
