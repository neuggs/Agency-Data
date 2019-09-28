library(xlsx)
library(caret)
library(dplyr)
library(gbm)          
library(xgboost)

# Read the original data set
agency_data_orig <- read.xlsx('./data/AgencyData_clean.xlsx', sheetIndex=1, stringsAsFactors=T)

# Trim the phat - i.e., data that's irrelevant
agency_data_used <- agency_data_orig[c(-1, -4)]
str(agency_data_used)

# Reusable function for one-hot encodigin
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


# Setup train and test data, 70% / 30%
train_ad <- sample_frac(agency_data_used, 0.7)
sid <- as.numeric(rownames(train_ad)) # because rownames() returns character
test_ad <- agency_data_used[-sid,]

fit <- glm(transaction_type ~ ., data=agency_data_used, family=poisson())
summary(fit) 

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = transaction_type ~ .,
  distribution = "gaussian",
  data = agency_data_used,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# print results
print(gbm.fit)

# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv")

set.seed(123)
# train GBM model
gbm.fit2 <- gbm(
  formula = transaction_type ~ .,
  distribution = "gaussian",
  data = agency_data_used,
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
) 
min_MSE <- which.min(gbm.fit2$cv.error)
sqrt(gbm.fit2$cv.error[min_MSE])
gbm.perf(gbm.fit2, method = "cv")
