
# Introduction

The data we will be analyzing in this project is a daily count of rental
bikes between years 2011 and 2012 in the Capital bikeshare system. This
bike share data set includes information about the day of rental and the
weather on that particular day. Below is a list of the variables that
will be available for us to include in our models and a brief
description:

  - `season` : season (1:winter, 2:spring, 3:summer, 4:fall)
  - `yr` : year (0: 2011, 1:2012)
  - `mnth` : month ( 1 to 12)
  - `holiday` : weather day is holiday or not
  - `weekday` : day of the week
  - `workingday` : if day is neither weekend nor holiday is 1, otherwise
    is 0
  - `weathersit` :
      - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
      - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
      - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds,
        Light Rain + Scattered clouds
      - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
  - `temp` : Normalized temperature in Celsius
  - `atemp`: Normalized feeling temperature in Celsius
  - `hum`: Normalized humidity
  - `windspeed`: Normalized wind speed
  - `cnt`: count of total rental bikes

The purpose of this analysis is to compare two models in terms of their
predictive performance. As this is a regression problem, we will use
RMSE to determine which model is the better fit. The models we will fit
are a non-ensemble tree model and a boosted tree model. Tuning
parameters for both models will be selected using leave one out cross
validation. We will fit both of these models on the training data set
and evaluate the RMSE on the test set.

# Loading Packages

We will load in our necessary packages, `tidyverse` and `caret`. We will
also set the seed, so our results are reproducible.

``` r
set.seed(123)
library(tidyverse)
library(caret)
library(ggplot2)
```

# Reading in Data

Using the `read_csv` function, we will read in the csv file of the bike
sharing data. With the use of the `select` function, we can remove the
`casual` and `registered` variables, which should not be used for
modeling, and any non-numeric variables, like `dteday`. Finally, using
`filter`, we will filter our data set by the specific day of the week we
are interested in analyzing for that report.

``` r
bikeData <- read_csv("day.csv")
bikeData <- bikeData %>% select(-c(casual, registered, instant, dteday)) %>% filter(weekday == params$dayofWeek)
```

# Creating Training and Test Split

Using `createDataPartition`, we will partition our data into the 70/30
training and test split.

``` r
bikeDataIndex <- createDataPartition(bikeData$cnt, p = 0.7, list = FALSE)
bikeDataTrain <- bikeData[bikeDataIndex, ]
bikeDataTest <- bikeData[-bikeDataIndex, ]
```

# Summarizations of Data

First, we will take a look at the five number summary of each variable
available in the data set.

``` r
summary(bikeDataTrain)
```

    ##      season            yr              mnth           holiday     weekday    workingday   weathersit   
    ##  Min.   :1.000   Min.   :0.0000   Min.   : 1.000   Min.   :0   Min.   :0   Min.   :0    Min.   :1.000  
    ##  1st Qu.:2.000   1st Qu.:0.0000   1st Qu.: 4.000   1st Qu.:0   1st Qu.:0   1st Qu.:0    1st Qu.:1.000  
    ##  Median :3.000   Median :1.0000   Median : 7.000   Median :0   Median :0   Median :0    Median :1.000  
    ##  Mean   :2.539   Mean   :0.5132   Mean   : 6.605   Mean   :0   Mean   :0   Mean   :0    Mean   :1.342  
    ##  3rd Qu.:4.000   3rd Qu.:1.0000   3rd Qu.: 9.250   3rd Qu.:0   3rd Qu.:0   3rd Qu.:0    3rd Qu.:2.000  
    ##  Max.   :4.000   Max.   :1.0000   Max.   :12.000   Max.   :0   Max.   :0   Max.   :0    Max.   :3.000  
    ##       temp            atemp             hum           windspeed            cnt      
    ##  Min.   :0.1667   Min.   :0.1616   Min.   :0.2758   Min.   :0.05038   Min.   : 605  
    ##  1st Qu.:0.3412   1st Qu.:0.3441   1st Qu.:0.5095   1st Qu.:0.13309   1st Qu.:2940  
    ##  Median :0.4612   Median :0.4561   Median :0.6629   Median :0.17755   Median :4358  
    ##  Mean   :0.4829   Mean   :0.4651   Mean   :0.6340   Mean   :0.18563   Mean   :4309  
    ##  3rd Qu.:0.6390   3rd Qu.:0.5979   3rd Qu.:0.7359   3rd Qu.:0.22575   3rd Qu.:5571  
    ##  Max.   :0.8058   Max.   :0.7311   Max.   :0.9483   Max.   :0.39801   Max.   :8227

From this output, we can see that the `yr`, `holiday`, and `workingday`
variables are binary, that is they take on values of 0 or 1. Also, the
`season`, `mnth`, and `weathersit` variables are categorical. We will
create contingency tables of these non-numeric variables and the count
of bikes shared. To create these tables, we will use the `aggregate`
function in combination with `kable`. First, we will create the table of
`year` and `cnt`.

``` r
knitr::kable(aggregate(bikeDataTrain$cnt, by = list(bikeDataTrain$yr), FUN = sum), col.names = c("Year", "Sum of Count"))
```

| Year | Sum of Count |
| ---: | -----------: |
|    0 |       126211 |
|    1 |       201271 |

From the table, we can see that the bike share rented out more bikes in
the year 2012, suggesting that the bike sharing company had better
performance in the year 2012. Secondly, we will look at a table of the
`holiday` and `cnt`.

``` r
knitr::kable(aggregate(bikeDataTrain$cnt, by = list(bikeDataTrain$holiday), FUN = sum), col.names = c("Holiday", "Sum of Count"))
```

| Holiday | Sum of Count |
| ------: | -----------: |
|       0 |       327482 |

As expected, this bike sharing company does more business on
non-holidays. This makes logical sense as there are more of these days
in a year than holidays. Next, we will look at the table of
`workingdays` and `cnt`.

``` r
knitr::kable(aggregate(bikeDataTrain$cnt, by = list(bikeDataTrain$workingday), FUN = sum), col.names = c("Working Day", "Sum of Count"))
```

| Working Day | Sum of Count |
| ----------: | -----------: |
|           0 |       327482 |

From the table, we can see that the count is higher for the weekdays,
rather than the weekends, this suggests that bike sharing may be
becoming a popular option for the work commute. The next table we will
create is of `season` and `cnt`.

``` r
knitr::kable(aggregate(bikeDataTrain$cnt, by = list(bikeDataTrain$season), FUN = sum), col.names = c("Season", "Sum of Count"))
```

| Season | Sum of Count |
| -----: | -----------: |
|      1 |        43288 |
|      2 |        92973 |
|      3 |       104782 |
|      4 |        86439 |

The most popular seasons appear to be summer and fall. And the least
popular season to utilize the bike share is winter. Next, we will look
at a table of `mnth` and `cnt`.

``` r
knitr::kable(aggregate(bikeDataTrain$cnt, by = list(bikeDataTrain$mnth), FUN = sum), col.names = c("Month", "Sum of Count"))
```

| Month | Sum of Count |
| ----: | -----------: |
|     1 |        13573 |
|     2 |        14828 |
|     3 |        19026 |
|     4 |        28568 |
|     5 |        35965 |
|     6 |        28642 |
|     7 |        39172 |
|     8 |        33839 |
|     9 |        44686 |
|    10 |        25423 |
|    11 |        27902 |
|    12 |        15858 |

We can see that the most popular months are those that fall in the
summer and fall seasons. The last contingency table we will create is
for `weather` and `cnt`.

``` r
knitr::kable(aggregate(bikeDataTrain$cnt, by = list(bikeDataTrain$weathersit), FUN = sum), col.names = c("Weather", "Sum of Count"))
```

| Weather | Sum of Count |
| ------: | -----------: |
|       1 |       225035 |
|       2 |       101420 |
|       3 |         1027 |

The bike share receives the most use when the weather is nice, with no
rain, snow, or thunderstorms. Now, we will create some histograms of the
remaining predictors and our reponse variable, `cnt`. We will create
these histograms using `ggplot` and `geom_jitter`. The first histogram
will contain our `temp` and `cnt` variables.

``` r
g <- ggplot(bikeDataTrain, aes(x = temp, y = cnt))
g + geom_jitter() + labs(x = "Normalized Temperature", y = "Count of Total Rental Bikes", title = "Temperature vs. Count")
```

![](Sunday_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

There is a clear positive trend in the histogram, as the temperature
becomes warmer, the number of rentals that day increases. The next
histogram we look at will contain the `atemp` and `cnt` variables.

``` r
g <- ggplot(bikeDataTrain, aes(x = atemp, y = cnt))
g + geom_jitter() + labs(x = "Normalized Feeling Temperature", y = "Count of Total Rental Bikes", title = "Feeling Temperature vs. Count")
```

![](Sunday_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

Much like the regular temperature, the temperature that it actually
feels like has a positive relationship with the number of rentals. Next,
we will create a histogram for the `hum` and `cnt` variables.

``` r
g <- ggplot(bikeDataTrain, aes(x = hum, y = cnt))
g + geom_jitter() + labs(x = "Normalized Humidity", y = "Count of Total Rental Bikes", title = "Humidity vs. Count")
```

![](Sunday_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

There doesn’t appear to be a definite relationship between the humidity
and the count of rental bikes. The final histogram will contain
`windspeed` and `cnt`.

``` r
g <- ggplot(bikeDataTrain, aes(x = windspeed, y = cnt))
g + geom_jitter() + labs(x = "Normalized Wind Speed", y = "Count of Total Rental Bikes", title = "Wind Speed vs. Count")
```

![](Sunday_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

There appears to be a slight negative relationship between wind speed
and the number of rentals that day. Next, we will move on to fitting our
models.

# Models

Now that we have read in our data, created our split, and done some
exploratory data analysis, we will begin fitting our models. The goal is
to create two models that predict the `cnt` variable in our data set.

## Non-ensemble Tree Model

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

``` r
(treeFit <- train(cnt ~ ., data = bikeDataTrain,
               method = "rpart",
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "LOOCV")))
```

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
    ##   cp          RMSE      Rsquared    MAE      
    ##   0.04033305  1150.910  0.62557671   911.3051
    ##   0.19759413  1427.802  0.43238556  1226.4758
    ##   0.55472332  1965.865  0.01058233  1784.7007
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.04033305.

The optimal model in this case used cp = 0.040333. And we can see the
training RMSE obtained in the output above.

## Boosted Tree Model

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

``` r
(boostedtreeFit <- train(cnt ~ ., data = bikeDataTrain,
               method = "gbm",
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "LOOCV"),
               verbose = FALSE))
```

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
    ##    50      1                  920.1846  0.7612359  732.3073
    ##    50      2                  876.4758  0.7783891  698.1969
    ##    50      3                  881.6316  0.7753884  719.2342
    ##   100      1                  868.4899  0.7818925  686.6267
    ##   100      2                  840.7523  0.7951453  670.8954
    ##   100      3                  880.3726  0.7757209  697.1818
    ##   150      1                  843.4098  0.7939069  675.6483
    ##   150      2                  851.1420  0.7901427  676.4245
    ##   150      3                  842.0850  0.7949724  655.8386
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode'
    ##  was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 100, interaction.depth = 2, shrinkage = 0.1
    ##  and n.minobsinnode = 10.

The optimal model in this case used n.trees = 100, interaction.depth =
2, shrinkage = 0.1, and n.minosbinnode = 10. And we can see the training
RMSE obtained in the output above.

# Testing Models

Now that we have determined the optimal fit of each model, we will apply
our models to the test set. First, we will obtain the test RMSE of the
tree model using `predict` and `postResample`.

``` r
treePred <- predict(treeFit, newdata = bikeDataTest)
(treeResults <- postResample(treePred, bikeDataTest$cnt))
```

    ##         RMSE     Rsquared          MAE 
    ## 1110.1503158    0.6466168  926.4468706

Again, we will use `predict` and `postResample` to obtain the test RMSE
of the boosted tree model.

``` r
boostedtreePred <- predict(boostedtreeFit, newdata = bikeDataTest)
(boostedtreeResults <- postResample(boostedtreePred, bikeDataTest$cnt))
```

    ##        RMSE    Rsquared         MAE 
    ## 817.3596274   0.8201213 643.4759122

The optimal model in this case is the boosted tree. And the test RMSE
was minimized at 817.3596274.

# Linear Regression Model

For the last model, a multiple linear regression model shall be added
along with the predictions for the model on the test set.

``` r
lm_model <- lm(cnt ~ ., data = bikeDataTest)
lm_model
```

    ## 
    ## Call:
    ## lm(formula = cnt ~ ., data = bikeDataTest)
    ## 
    ## Coefficients:
    ## (Intercept)       season           yr         mnth      holiday      weekday   workingday   weathersit  
    ##      1707.3        752.9       1536.3       -129.7           NA           NA           NA      -1421.3  
    ##        temp        atemp          hum    windspeed  
    ##      8136.2      -4369.7       1248.7      -1846.2

Now we will predict the values for the multiple linear regression model
along with the confidence intervals

``` r
lm_model_predict <- predict(lm_model, interval = "confidence")
lm_model_predict
```

    ##         fit         lwr      upr
    ## 1  1309.894 -135.967197 2755.754
    ## 2  1400.985   -6.757441 2808.728
    ## 3  1352.450  148.932185 2555.967
    ## 4  2278.433  956.322448 3600.543
    ## 5  2139.663 1223.785639 3055.541
    ## 6  2958.953 1983.795146 3934.111
    ## 7  3079.404 2022.299044 4136.509
    ## 8  2830.828 1520.860845 4140.795
    ## 9  4511.680 2872.551455 6150.808
    ## 10 4979.133 3642.712415 6315.554
    ## 11 5291.936 4143.169466 6440.702
    ## 12 4990.508 3722.032353 6258.983
    ## 13 4784.727 3533.079038 6036.376
    ## 14 4902.186 3366.181270 6438.191
    ## 15 4052.221 2559.939178 5544.503
    ## 16 2263.059  725.342907 3800.774
    ## 17 2733.526 1108.624227 4358.428
    ## 18 3742.157 2546.631880 4937.681
    ## 19 5371.582 4370.030102 6373.135
    ## 20 5987.935 4669.176321 7306.694
    ## 21 5351.134 4317.403090 6384.864
    ## 22 6867.195 5573.583542 8160.807
    ## 23 6895.547 5590.526628 8200.568
    ## 24 5223.751 3757.585801 6689.917
    ## 25 5845.493 4506.646473 7184.340
    ## 26 4952.594 3638.719230 6266.469
    ## 27 4115.518 2748.154288 5482.882
    ## 28 4049.950 2705.556272 5394.344
    ## 29 2282.558   23.966054 4541.150
