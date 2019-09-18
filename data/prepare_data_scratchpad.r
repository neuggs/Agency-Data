# Scratchpad for prepare data.
library(readxl)

agency_data_orig <- read_excel('./data/AgencyData_clean.xlsx')

# Variable by variable analysis / preparation
# account_name
# Each instance is unique, which is expected given it's the insured name. This variable is not 
# useful and should be removed.
agency_data <- agency_data_orig[-1]

# account_type
# This is a two-factor variable containing `Personal` or `Commercial` - the former far more
# frequently. There is no ordering to these values