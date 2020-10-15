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

    ## Classes 'tbl_df', 'tbl' and 'data.frame':    105 obs. of  14 variables:
    ##  $ instant   : num  3 10 17 24 31 38 45 52 59 66 ...
    ##  $ dteday    : Date, format: "2011-01-03" ...
    ##  $ season    : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ mnth      : num  1 1 1 1 1 2 2 2 2 3 ...
    ##  $ holiday   : num  0 0 1 0 0 0 0 1 0 0 ...
    ##  $ weekday   : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ workingday: num  1 1 0 1 1 1 1 0 1 1 ...
    ##  $ weathersit: num  1 1 2 1 2 1 1 2 2 1 ...
    ##  $ temp      : num  0.1964 0.1508 0.1758 0.0974 0.1808 ...
    ##  $ atemp     : num  0.189 0.151 0.177 0.118 0.186 ...
    ##  $ hum       : num  0.437 0.483 0.537 0.492 0.604 ...
    ##  $ windspeed : num  0.248 0.223 0.194 0.158 0.187 ...
    ##  $ cnt       : num  1349 1321 1000 1416 1501 ...

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
    ##   cp         RMSE      Rsquared    MAE     
    ##   0.1227148  1567.676  0.27548702  1175.554
    ##   0.1359665  1625.937  0.19910766  1333.294
    ##   0.4184550  2077.472  0.07279865  1844.919
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.1227148.

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
    ##    50      1                  985.7200  0.6876703  623.1432
    ##    50      2                  977.1974  0.6921589  634.4080
    ##    50      3                  984.9771  0.6882257  636.9509
    ##   100      1                  948.6733  0.7094962  587.3141
    ##   100      2                  929.1957  0.7214639  580.1172
    ##   100      3                  930.6775  0.7212525  588.2118
    ##   150      1                  937.0247  0.7167968  572.5697
    ##   150      2                  885.2588  0.7472324  543.7724
    ##   150      3                  895.3479  0.7417785  548.8530
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of
    ##  0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a
    ##  value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees =
    ##  150, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

    treePred <- predict(treeFit, newdata = bikeDataTest)
    postResample(treePred, bikeDataTest$cnt)

    ##        RMSE    Rsquared         MAE 
    ## 1299.278206    0.509443  907.031130

    boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
    postResample(boostedtreePred, bikeDataTest$cnt)

    ##        RMSE    Rsquared         MAE 
    ## 646.7390129   0.8974421 490.2841328
