import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pickle

import warnings
warnings.filterwarnings('ignore')

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
agency_df_used = ohe(agency_df_used, 'transaction_type')

# Simple replace for ordinal value
policy_term_mapper = {"6 Months": 1, "12 Months":2}
agency_df_used.replace(policy_term_mapper, inplace=True)

# Convert the time
agency_df_used['eff_date_int'] = pd.to_datetime(agency_df_used['effective_date']).astype(np.int64)
agency_df_used.drop(['effective_date'],axis=1, inplace=True)

# Set features and target
target = agency_df_used['transaction_type_New Business']
print(agency_df_used['transaction_type_New Business'])
features = agency_df_used.loc[:, agency_df_used.columns != 'transaction_type_New Business']

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=1)

random_grid = {'min_child_weight': [1, 5, 1, 0],
               'gamma': [0.5, 1, 1.5, 2, 5],
               'subsample': [0.6, 0.8, 1.0],
               'colsample_bytree': [0.6, 0.8, 1.0],
               'min_depth': [3, 4, 5]}

def xgbcClassifier(the_grid):
    xgbc_classifier = XGBClassifier(learning_rate=0.02, n_estimator=600, objective='binary:logistic',
                                    silent=True, nthread=1)
    folds = 3
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    random_search = RandomizedSearchCV(xgbc_classifier, param_distributions=the_grid,
                                       n_iter=param_comb, scoring='roc_auc', n_jobs=4,
                                       cv=skf.split(features_train, target_train), verbose=3,
                                       random_state=1001)
    random_search.fit(features_train, target_train)
    return random_search

if __name__ == '__main__':
    random_search_model = xgbcClassifier(random_grid)
    print("\n Best estimator:", random_search_model.best_estimator_)
    print("Best normalized gini scores:", random_search_model.best_score_)
    print("Best hyperparameters:", random_search_model.best_params_)

    # Save the model
    pkl_filename = "./model/XGB_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(random_search_model, file)










