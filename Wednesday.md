Report for One Day of Week
================
Rachel Keller
October 16, 2020

    set.seed(123)
    library(tidyverse)
    library(caret)

Group B Data:

    bikeData <- read_csv("day.csv")
    bikeData$dteday <- as.Date(bikeData$dteday, format = "%m/%d/%Y")
    bikeData <- bikeData %>% select(-c(casual, registered)) %>% filter(weekday == params$dayofWeek)
    str(bikeData)

    ## Classes 'tbl_df', 'tbl' and 'data.frame':    104 obs. of  14 variables:
    ##  $ instant   : num  5 12 19 26 33 40 47 54 61 68 ...
    ##  $ dteday    : Date, format: "2011-01-05" ...
    ##  $ season    : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ mnth      : num  1 1 1 1 2 2 2 2 3 3 ...
    ##  $ holiday   : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday   : num  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ workingday: num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weathersit: num  1 1 2 3 2 2 1 1 1 2 ...
    ##  $ temp      : num  0.227 0.173 0.292 0.217 0.26 ...
    ##  $ atemp     : num  0.229 0.16 0.298 0.204 0.254 ...
    ##  $ hum       : num  0.437 0.6 0.742 0.863 0.775 ...
    ##  $ windspeed : num  0.187 0.305 0.208 0.294 0.264 ...
    ##  $ cnt       : num  1600 1162 1650 506 1526 ...

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
    ##   cp          RMSE      Rsquared    MAE     
    ##   0.06556674  1235.807  0.61399265  1016.602
    ##   0.11958555  1412.293  0.49689172  1271.758
    ##   0.62485106  2092.887  0.01465002  1929.871
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.06556674.

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
    ##    50      1                  801.6425  0.8348517  607.2494
    ##    50      2                  810.0885  0.8298018  603.9451
    ##    50      3                  793.9236  0.8366740  615.0908
    ##   100      1                  789.0940  0.8386876  580.8118
    ##   100      2                  829.3827  0.8227369  616.2675
    ##   100      3                  788.0567  0.8391429  600.0130
    ##   150      1                  802.9909  0.8334840  587.1190
    ##   150      2                  834.2969  0.8205621  615.4377
    ##   150      3                  792.9991  0.8370592  599.5326
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of
    ##  0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a
    ##  value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees =
    ##  100, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode
    ##  = 10.

    treePred <- predict(treeFit, newdata = bikeDataTest)
    postResample(treePred, bikeDataTest$cnt)

    ##         RMSE     Rsquared          MAE 
    ## 1556.3364828    0.5253235 1102.0804243

    boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
    postResample(boostedtreePred, bikeDataTest$cnt)

    ##         RMSE     Rsquared          MAE 
    ## 1016.3677219    0.7999325  717.3686499
