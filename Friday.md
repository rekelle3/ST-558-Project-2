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
    ##  $ instant   : num  7 14 21 28 35 42 49 56 63 70 ...
    ##  $ dteday    : Date, format: "2011-01-07" ...
    ##  $ season    : num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ yr        : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ mnth      : num  1 1 1 1 2 2 2 2 3 3 ...
    ##  $ holiday   : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday   : num  5 5 5 5 5 5 5 5 5 5 ...
    ##  $ workingday: num  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weathersit: num  2 1 1 2 2 1 1 2 2 2 ...
    ##  $ temp      : num  0.197 0.161 0.177 0.203 0.211 ...
    ##  $ atemp     : num  0.209 0.188 0.158 0.223 0.229 ...
    ##  $ hum       : num  0.499 0.538 0.457 0.793 0.585 ...
    ##  $ windspeed : num  0.169 0.127 0.353 0.123 0.128 ...
    ##  $ cnt       : num  1510 1421 1543 1167 1708 ...

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
    ##   0.05923083  1054.666  0.66407826   853.966
    ##   0.15806133  1278.713  0.50024066  1115.693
    ##   0.57006022  1908.218  0.01102411  1737.565
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.05923083.

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
    ##    50      1                  850.3475  0.7809338  673.2409
    ##    50      2                  846.9692  0.7796530  653.1450
    ##    50      3                  839.5665  0.7833560  654.2296
    ##   100      1                  819.0775  0.7935573  633.8274
    ##   100      2                  835.0596  0.7855509  647.7357
    ##   100      3                  846.9224  0.7800547  647.3957
    ##   150      1                  815.8363  0.7951663  630.7458
    ##   150      2                  833.8810  0.7863438  651.4733
    ##   150      3                  841.2959  0.7829176  638.3438
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of
    ##  0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a
    ##  value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees =
    ##  150, interaction.depth = 1, shrinkage = 0.1 and n.minobsinnode
    ##  = 10.

    treePred <- predict(treeFit, newdata = bikeDataTest)
    postResample(treePred, bikeDataTest$cnt)

    ##         RMSE     Rsquared          MAE 
    ## 1152.0163326    0.6713208  915.9265873

    boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
    postResample(boostedtreePred, bikeDataTest$cnt)

    ##        RMSE    Rsquared         MAE 
    ## 917.8024817   0.8069202 772.1393691
