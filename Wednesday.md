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

The purpose of this analysis is to compare two models in terms of their
predictive performance. As this is a regression problem, we will use
RMSE to determine which model is the better fit. The models we will fit
are a non-ensemble tree model and a boosted tree model. Tuning
parameters for both models will be selected using leave one out cross
validation. We will fit both of these models on the training data set
and evaluate the RMSE on the test set.

Set Up and Required Packages
============================

We will load in our necessary packages, `tidyverse` and `caret`. We will
also set the seed, so our results are reproducible.

    set.seed(123)
    library(tidyverse)
    library(caret)

Reading in Data
===============

Using the `read_csv` function, we will read in the csv file of the bike
sharing data. With the use of the `select` function, we can remove the
casual and registered variables, which should not be used for modeling,
and any non-numeric variables, like dteday. Finally, using `filter`, we
will filter our data set by the specific day of the week we are
interested in analyzing for that report.

    bikeData <- read_csv("day.csv")
    bikeData <- bikeData %>% select(-c(casual, registered, instant, dteday)) %>% filter(weekday == params$dayofWeek)

Creating Training and Test Split
================================

Using `createDataPartition`, we will partition our data into the 70/30
training and test split.

    bikeDataIndex <- createDataPartition(bikeData$cnt, p = 0.7, list = FALSE)
    bikeDataTrain <- bikeData[bikeDataIndex, ]
    bikeDataTest <- bikeData[-bikeDataIndex, ]

Summarizations of Data
======================

    summary(bikeDataTrain)

    ##      season            yr              mnth          holiday       
    ##  Min.   :1.000   Min.   :0.0000   Min.   : 1.00   Min.   :0.00000  
    ##  1st Qu.:2.000   1st Qu.:0.0000   1st Qu.: 3.75   1st Qu.:0.00000  
    ##  Median :3.000   Median :0.0000   Median : 7.00   Median :0.00000  
    ##  Mean   :2.539   Mean   :0.4737   Mean   : 6.50   Mean   :0.01316  
    ##  3rd Qu.:3.000   3rd Qu.:1.0000   3rd Qu.: 9.00   3rd Qu.:0.00000  
    ##  Max.   :4.000   Max.   :1.0000   Max.   :12.00   Max.   :1.00000  
    ##     weekday    workingday       weathersit         temp       
    ##  Min.   :3   Min.   :0.0000   Min.   :1.000   Min.   :0.1075  
    ##  1st Qu.:3   1st Qu.:1.0000   1st Qu.:1.000   1st Qu.:0.3486  
    ##  Median :3   Median :1.0000   Median :1.000   Median :0.5400  
    ##  Mean   :3   Mean   :0.9868   Mean   :1.447   Mean   :0.5181  
    ##  3rd Qu.:3   3rd Qu.:1.0000   3rd Qu.:2.000   3rd Qu.:0.6690  
    ##  Max.   :3   Max.   :1.0000   Max.   :3.000   Max.   :0.7933  
    ##      atemp             hum           windspeed            cnt      
    ##  Min.   :0.1193   Min.   :0.4029   Min.   :0.06096   Min.   : 705  
    ##  1st Qu.:0.3508   1st Qu.:0.5640   1st Qu.:0.12904   1st Qu.:2653  
    ##  Median :0.5205   Median :0.6415   Median :0.16884   Median :4642  
    ##  Mean   :0.4938   Mean   :0.6547   Mean   :0.18297   Mean   :4534  
    ##  3rd Qu.:0.6231   3rd Qu.:0.7476   3rd Qu.:0.23431   3rd Qu.:5846  
    ##  Max.   :0.7469   Max.   :0.9704   Max.   :0.41543   Max.   :8173

    corrplot::corrplot(cor(bikeDataTrain))

    ## Warning in cor(bikeDataTrain): the standard deviation is zero

![](Wednesday_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Models
======

Now that we have read in our data, created our split, and done some
exploratory data analysis, we will begin fitting our models. The goal is
to create two models that predict the cnt variable in our data set.

Nonensemble Tree Model
----------------------

The first model we will fit is a regression tree. The main idea of this
model is to split up our predictor space into regions, and for a given
region, use the main of the observations as our predictor value. For the
fitting process of this model, we will use leave one out cross
validation. For LOOCV, one observation is removed and the model is fit
on the remaining data, and this fit is used to predict the value of the
deleted observation. We repeated this process for each observation and
compute the mean square error. The data was also centered and scaled
using the `preProcess` function. The final choosen model will be the one
that minimzes the training RMSE. For the tuning parameter of cp, we will
use the default values rather than providing a grid of tuning
parameters.

    (treeFit <- train(cnt ~ ., data = bikeDataTrain,
                   method = "rpart",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "LOOCV")))

    ## CART 
    ## 
    ## 76 samples
    ## 11 predictors
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 75, 75, 75, 75, 75, 75, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp         RMSE      Rsquared    MAE     
    ##   0.1007572  1328.191  0.54735171  1088.520
    ##   0.2437851  1727.282  0.26296344  1496.171
    ##   0.4198293  2321.965  0.03985709  2120.212
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.1007572.

The optimal model in this case used cp = 0.1007572. And we can see the
training RMSE obtained in the output above.

Boosted Tree Model
------------------

The final model we will fit is a boosted tree. This model builds off of
the previous in that we are sequentially fitting tree models. Each
subsequent tree is grown on a modified version of the training data, and
we update our predictions as the tree grows. For the fitting process of
this model, we will use leave one out cross validation. For LOOCV, one
observation is removed and the model is fit on the remaining data, and
this fit is used to predict the value of the deleted observation. We
repeated this process for each observation and compute the mean square
error. The data was also centered and scaled using the `preProcess`
function. The final choosen model will be the one that minimzes the
training RMSE. For the tuning parameters of number of trees, depth,
shrinkage, and minimum number of observations in a node, we will use the
default values rather than providing a grid of tuning parameters.

    (boostedtreeFit <- train(cnt ~ ., data = bikeDataTrain,
                   method = "gbm",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "LOOCV"),
                   verbose = FALSE))

    ## Stochastic Gradient Boosting 
    ## 
    ## 76 samples
    ## 11 predictors
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 75, 75, 75, 75, 75, 75, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   n.trees  interaction.depth  RMSE      Rsquared   MAE     
    ##    50      1                  848.0970  0.8191446  690.4365
    ##    50      2                  775.0494  0.8494568  592.9884
    ##    50      3                  812.9631  0.8306941  634.8089
    ##   100      1                  800.1539  0.8342576  642.5959
    ##   100      2                  757.4757  0.8518330  567.2268
    ##   100      3                  781.7939  0.8414513  602.9526
    ##   150      1                  801.8680  0.8332930  634.2291
    ##   150      2                  763.5737  0.8489750  568.9567
    ##   150      3                  780.0101  0.8422437  588.1370
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of
    ##  0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a
    ##  value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees =
    ##  100, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

The optimal model in this case used n.trees = 100, interaction.depth =
2, shrinkage = 0.1, and n.minosbinnode = 10. And we can see the training
RMSE obtained in the output above.

Testing Models on Test Set
==========================

Now that we have determined the optimal fit of each model, we will apply
our models to the test set. First, we will obtain the test RMSE of the
tree model using `predict` and `postResample`.

    treePred <- predict(treeFit, newdata = bikeDataTest)
    (treeResults <- postResample(treePred, bikeDataTest$cnt))

    ##         RMSE     Rsquared          MAE 
    ## 1648.4413511    0.4444568 1338.8170937

Again, we will use `predict` and `postResample` to obtain the test RMSE
of the boosted tree model.

    boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
    (boostedtreeResults <- postResample(boostedtreePred, bikeDataTest$cnt))

    ##        RMSE    Rsquared         MAE 
    ## 846.6171247   0.8758602 684.8564741

The optimal model in this case is the boosted tree.
