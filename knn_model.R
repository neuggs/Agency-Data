library(xlsx)
library(caret)
library(dplyr)
library(class)

source('load_trim_data.R')

# Reusable function for one-hot encoding
ordinal <- function(the_var, the_var_str){
  dmy <- dummyVars(paste(" ~ ", the_var_str), data = agency_data_used)
  dummy_df <- data.frame(predict(dmy, newdata = agency_data_used))
  agency_data_used <- cbind(agency_data_used, dummy_df)
  return(agency_data_used)
}

# Reusable function for non-ordingal
non_ordinal <- function(the_var){
  the_var <- as.integer(the_var)
  return (the_var)
}

nor <-function(x) { 
  (x -min(x))/(max(x)-min(x)) 
}

# Convert date to numeric
agency_data_used$effective_date <- as.numeric(agency_data_used$effective_date)

# Convert all the categorical features to numeric. Simple translation is required for 
# non-ordinal categorical variables. Ordinal categorical variables are encoded using 
# one hot encoding. 
agency_data_used$account_type <- non_ordinal(agency_data_used$account_type)
agency_data_used$assigned_agent <- non_ordinal(agency_data_used$assigned_agent)
agency_data_used$lob <- non_ordinal(agency_data_used$lob)
agency_data_used$master_company <- non_ordinal(agency_data_used$master_company)
agency_data_used$policy_type <- non_ordinal(agency_data_used$policy_type)
agency_data_used <- ordinal(agency_data_used$policy_term, "policy_term")
agency_data_used$policy_term <- NULL # can't get this into the function for some reason
agency_data_used$rating_state <- non_ordinal(agency_data_used$rating_state)
agency_data_used$status <- non_ordinal(agency_data_used$status)
agency_data_used$transaction_type <- non_ordinal(agency_data_used$transaction_type)

agency_data_used_norm <- as.data.frame(lapply(agency_data_used[,c(1,2,3,4, 5, 6, 7, 8,
                                                                  9, 10)], nor))
summary(agency_data_used_norm)

set.seed(123)
sample_norm <- sample(1:nrow(agency_data_used_norm),
                      size=nrow(agency_data_used_norm)*0.7,replace = FALSE)
train_ad <- agency_data_used[sample_norm,]
test_ad <- agency_data_used[-sample_norm,]
train_ad_target <- agency_data_used[sample_norm, 11]
test_ad_target <- agency_data_used[-sample_norm, 11]

sqrt(NROW(train_ad_target))
knn.40 <- knn(train=train_ad, test=test_ad, cl=train_ad_target, k=40)
knn.41 <- knn(train=train_ad, test=test_ad, cl=train_ad_target, k=41)

ACC.40 <- 100 * sum(test_ad_target == knn.40)/NROW(test_ad_target)
ACC.41 <- 100 * sum(test_ad_target == knn.41)/NROW(test_ad_target)
print(paste("K=40 accuracy:", ACC.40))
print(paste("k=41 accurac:", ACC.41))
table(knn.40 ,test_ad_target)
table(knn.41 ,test_ad_target)

#confusionMatrix(table(knn.40, test_ad_target))
i=1
k.optm=1
for (i in 1:28){
  knn.mod <- knn(train=train_ad, test=test_ad, cl=train_ad_target, k=i)
  k.optm[i] <- 100 * sum(test_ad_target == knn.mod)/NROW(test_ad_target)
  k=i
  cat(k,'=',k.optm[i],'')
}
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")
