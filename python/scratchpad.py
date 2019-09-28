import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import numpy as np
import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

agency_df = pd.read_excel("../data/AgencyData_clean.xlsx")
print(agency_df.shape)

# Trim the phat
agency_df_used = agency_df.drop(['account_name', 'branch_name'], axis=1)
print(agency_df_used.head(2))
print(agency_df_used.shape)

"""
features = agency_df_used[[
    "account_type",
    "assigned_agent",
    "lob",
    "master_company",
    "effective_date",
    "policy_term",
    "policy_type",
    "annual_premium",
    "written_premium",
    "rating_state",
    "status"
]]
"""

def ohe(df, feature):
    # encode
    df = pd.concat([df,pd.get_dummies(df[feature], prefix=feature)],axis=1)
    # now drop the field, it's no longer needed
    df.drop([feature],axis=1, inplace=True)
    return df

agency_df_used = ohe(agency_df_used, 'account_type')
agency_df_used = ohe(agency_df_used, 'assigned_agent')
agency_df_used = ohe(agency_df_used, 'lob')
agency_df_used = ohe(agency_df_used, 'master_company')
agency_df_used = ohe(agency_df_used, 'policy_type')
agency_df_used = ohe(agency_df_used, 'rating_state')
agency_df_used = ohe(agency_df_used, 'status')

# Simple replace for ordinal value
policy_term_mapper = {"6 Months": 1, "12 Months":2}
agency_df_used.replace(policy_term_mapper, inplace=True)

# Convert the time
agency_df_used['eff_date_int'] = pd.to_datetime(agency_df_used['effective_date']).astype(np.int64)
agency_df_used.drop(['effective_date'],axis=1, inplace=True)

# Set features and target
target = agency_df_used['transaction_type']
features = agency_df_used.loc[:, agency_df_used.columns != 'transaction_type']

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

decision_tree_classifier = DecisionTreeClassifier(random_state=0)
model = decision_tree_classifier.fit(features_train, target_train)

y_predictions = model.predict(features_test)
accuracy = accuracy_score(target_test, y_predictions)
accuracy_converted = round((accuracy * 100), 2)
print(accuracy_converted, "%")








