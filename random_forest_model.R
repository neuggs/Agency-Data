library(xlsx)
library(caret)
library(randomForest)


# Read the original data set
agency_data_orig <- read.xlsx('./data/AgencyData_clean.xlsx', sheetIndex=1, stringsAsFactors=T)

str(agency_data_orig)
# Trim the phat - i.e., data that's irrelevant
agency_data_used <- agency_data_orig[c(-1, -4)]
str(agency_data_used)

set.seed(100)
sample_ad <- sample(nrow(agency_data_used), 0.7*nrow(agency_data_used), replace = FALSE)
train_ad <- agency_data_used[sample_ad,]
train_ad$transaction_type <- factor(train_ad$transaction_type)
test_ad <- agency_data_used[-sample_ad,]
test_ad$transaction_type <- factor(test_ad$transaction_type)

rf_model <- randomForest(transaction_type ~ ., data = train_ad, ntree=700, importance = TRUE, do.trace=100)

