# Read the original data set
agency_data_orig <- read.xlsx('./data/AgencyData_clean.xlsx', sheetIndex=1, stringsAsFactors=T)

# Trim the phat - i.e., data that's irrelevant
agency_data_used <- agency_data_orig[c(-1, -4)]
str(agency_data_used)