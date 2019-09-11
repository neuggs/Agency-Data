# KNN for premium and dates
# NOTE: the variables from the naive_bayes_model must be in memory for this to 
# work correctly.
library(sjPlot)
library(class)
library(dplyr)

sjc.elbow(data.frame(test_ad_knn$written_premium))

train_prem <- data.frame(train_ad$written_premium)
test_prem <- data.frame(test_ad$written_premium)

knn_written_premium <- knn(train_prem, 
                           test_prem, 
                           cl=train_prem, k=4)
