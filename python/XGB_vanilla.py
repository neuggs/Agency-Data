import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

agency_df = pd.read_excel("../data/AgencyData_clean.xlsx")
print("Original agency df shape:", agency_df.shape)

# Trim the phat
agency_df_pruned = agency_df.drop(['account_name', 'branch_name'], axis=1)
print("Agency df pruned shape:", agency_df_pruned.shape)

# Need to remove values for transaction_type that have too few instances
agency_df_used = agency_df_pruned.groupby('transaction_type').filter(lambda x : len(x)>3)
print("Agency data used shape:", agency_df_used.shape)

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

agency_df_used.to_csv('./data/agency_df_used.csv')

# Set features and target
target = agency_df_used['transaction_type']
features = agency_df_used.loc[:, agency_df_used.columns != 'transaction_type']

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=1)

def xgb_vanilla():
    # Vanilla (bland) DTC with no tuning
    dt_class_bland = DecisionTreeClassifier(random_state=0)
    dt_class_bland.fit(features_train, target_train)
    y_predict = dt_class_bland.predict(features_test)
    acc = accuracy_score(target_test, y_predict)
    print("Bland accuracy score:", acc)

    return dt_class_bland

if __name__ == '__main__':
    rxg_bland_model = xgb_vanilla()

    # Save model model
    pkl_filename = "./model/XGB_vanilla_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(rxg_bland_model, file)









