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
    ##  $ instant   : num  4 11 18 25 32 39 46 53 60 67 ...
    ##  $ dteday    : Date, format: "2011-01-04" ...
    ##  $ season    : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ mnth      : num  1 1 1 1 2 2 2 2 3 3 ...
    ##  $ holiday   : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday   : num  2 2 2 2 2 2 2 2 2 2 ...
    ##  $ workingday: num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weathersit: num  1 2 2 2 2 1 1 1 1 1 ...
    ##  $ temp      : num  0.2 0.169 0.217 0.223 0.192 ...
    ##  $ atemp     : num  0.212 0.191 0.232 0.235 0.235 ...
    ##  $ hum       : num  0.59 0.686 0.862 0.617 0.83 ...
    ##  $ windspeed : num  0.1603 0.1221 0.1468 0.1298 0.0532 ...
    ##  $ cnt       : num  1562 1263 683 1985 1360 ...

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
    ##   cp          RMSE      Rsquared     MAE      
    ##   0.08618673  1125.992  0.581475785   810.5528
    ##   0.20031060  1354.517  0.387801426  1123.0533
    ##   0.46001019  1908.099  0.009087967  1709.5209
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.08618673.

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
    ##    50      1                  773.2987  0.8135941  560.1858
    ##    50      2                  715.9256  0.8317464  496.2478
    ##    50      3                  698.6762  0.8399822  497.4802
    ##   100      1                  701.3714  0.8359986  506.1137
    ##   100      2                  686.2799  0.8416051  460.9680
    ##   100      3                  677.7051  0.8457428  486.8399
    ##   150      1                  696.6375  0.8368503  501.8685
    ##   150      2                  674.5047  0.8466604  465.2146
    ##   150      3                  683.0116  0.8430771  492.0030
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

    ##         RMSE     Rsquared          MAE 
    ## 1474.4104405    0.4893503 1037.6037219

    boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
    postResample(boostedtreePred, bikeDataTest$cnt)

    ##        RMSE    Rsquared         MAE 
    ## 892.9212499   0.8271804 724.4013281
