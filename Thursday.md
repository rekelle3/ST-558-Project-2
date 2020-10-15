Report for One Day of Week
================
Rachel Keller
October 16, 2020

Introduction
============

The data we will be analyzing in this project is a daily count of rental
bikes between years 2011 and 2012 in the Capital bikeshare system. This
bike share data set includes information about the day of rental and the
weather on that particular day. Below is a list of the variables that
will be available for us to include in our models and a brief
description:

-   instant: record index
-   dteday : date
-   season : season (1:winter, 2:spring, 3:summer, 4:fall)
-   yr : year (0: 2011, 1:2012)
-   mnth : month ( 1 to 12)
-   hr : hour (0 to 23)
-   holiday : weather day is holiday or not
-   weekday : day of the week
-   workingday : if day is neither weekend nor holiday is 1, otherwise
    is 0
-   weathersit :
    -   1: Clear, Few clouds, Partly cloudy, Partly cloudy
    -   2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    -   3: Light Snow, Light Rain + Thunderstorm + Scattered clouds,
        Light Rain + Scattered clouds
    -   4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
-   temp : Normalized temperature in Celsius
-   atemp: Normalized feeling temperature in Celsius
-   hum: Normalized humidity
-   windspeed: Normalized wind speed
-   cnt: count of total rental bikes

<!-- -->

    set.seed(123)
    library(tidyverse)
    library(caret)

Group B Data:

    bikeData <- read_csv("day.csv")
    bikeData$dteday <- as.Date(bikeData$dteday, format = "%m/%d/%Y")
    bikeData <- bikeData %>% select(-c(casual, registered)) %>% filter(weekday == params$dayofWeek)
    str(bikeData)

    ## Classes 'tbl_df', 'tbl' and 'data.frame':    104 obs. of  14 variables:
    ##  $ instant   : num  6 13 20 27 34 41 48 55 62 69 ...
    ##  $ dteday    : Date, format: "2011-01-06" ...
    ##  $ season    : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ mnth      : num  1 1 1 1 2 2 2 2 3 3 ...
    ##  $ holiday   : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday   : num  4 4 4 4 4 4 4 4 4 4 ...
    ##  $ workingday: num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weathersit: num  1 1 2 1 1 1 1 2 1 3 ...
    ##  $ temp      : num  0.204 0.165 0.262 0.195 0.187 ...
    ##  $ atemp     : num  0.233 0.151 0.255 0.22 0.178 ...
    ##  $ hum       : num  0.518 0.47 0.538 0.688 0.438 ...
    ##  $ windspeed : num  0.0896 0.301 0.1959 0.1138 0.2778 ...
    ##  $ cnt       : num  1606 1406 1927 431 1550 ...

    bikeDataIndex <- createDataPartition(bikeData$cnt, p = 0.7, list = FALSE)
    bikeDataTrain <- bikeData[bikeDataIndex, ]
    bikeDataTest <- bikeData[-bikeDataIndex, ]

    (treeFit <- train(cnt ~ ., data = bikeDataTrain,
                   method = "rpart",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "LOOCV")))

    ## CART 
    ## 
    ## 76 samples
    ## 13 predictors
    ## 
    ## Pre-processing: centered (13), scaled (13) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 75, 75, 75, 75, 75, 75, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          RMSE      Rsquared      MAE      
    ##   0.09157769  1200.136  0.5986147905   953.0409
    ##   0.12494142  1297.275  0.5275444346  1092.0602
    ##   0.58934191  2040.661  0.0008472206  1864.5009
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.09157769.

    (boostedtreeFit <- train(cnt ~ ., data = bikeDataTrain,
                   method = "gbm",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "LOOCV"),
                   verbose = FALSE))

    ## Stochastic Gradient Boosting 
    ## 
    ## 76 samples
    ## 13 predictors
    ## 
    ## Pre-processing: centered (13), scaled (13) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 75, 75, 75, 75, 75, 75, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   n.trees  interaction.depth  RMSE      Rsquared   MAE     
    ##    50      1                  855.2634  0.7936390  640.5333
    ##    50      2                  855.8264  0.7933832  614.0520
    ##    50      3                  844.2467  0.7988057  631.4609
    ##   100      1                  864.5511  0.7899633  636.0754
    ##   100      2                  862.0618  0.7913718  621.3877
    ##   100      3                  846.3655  0.7982630  628.9764
    ##   150      1                  877.5151  0.7846950  645.7317
    ##   150      2                  849.7281  0.7973400  610.7626
    ##   150      3                  843.7464  0.7995048  643.5269
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of
    ##  0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a
    ##  value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees =
    ##  150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

    treePred <- predict(treeFit, newdata = bikeDataTest)
    postResample(treePred, bikeDataTest$cnt)

    ##         RMSE     Rsquared          MAE 
    ## 1142.8221999    0.6897695  827.5277778

    boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
    postResample(boostedtreePred, bikeDataTest$cnt)

    ##         RMSE     Rsquared          MAE 
    ## 1002.9416201    0.7659882  768.0420664
