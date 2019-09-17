library(xlsx)
#library(dplyr)
#l#ibrary(e1071)
#library(caret)
#library(formattable)

# Let's try Naive Bayes as-is with the important variables. This is admittedly a lazy
# attempt just to see what happens since Naive Bayes can handle categorical data.

# Read the original data set
agency_data_orig <- read.xlsx('./data/AgencyData_clean.xlsx', sheetIndex=1, stringsAsFactors=T)

# Trim the phat - i.e., data that's irrelevant
agency_data_used <- agency_data_orig[c(-1, -4)]

hist(agency_data_used$annual_premium)
scatter.smooth(agency_data_used$annual_premium)
hist(agency_data_used$written_premium)
scatter.smooth(agency_data_used$written_premium)

# transform written and annual premium
agency_data_used['sqrt_wp'] <- sqrt(agency_data_used$written_premium)
hist(agency_data_used$sqrt_wp)

agency_data_used['sqrt_ap'] <- sqrt(agency_data_used$annual_premium)
hist(agency_data_used$sqrt_ap)

agency_data_used['log10_wp'] <- log10((1 + agency_data_used$written_premium))
hist(agency_data_used$log10_wp)
scatter.smooth(agency_data_used$log10_wp)

agency_data_used['log10_ap'] <- log10((1 + agency_data_used$annual_premium))
hist(agency_data_used$log10_ap)
scatter.smooth(agency_data_used$log10_ap)
     