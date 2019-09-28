import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Set features and target
target = agency_df_used['transaction_type']
features = agency_df_used.loc[:, agency_df_used.columns != 'transaction_type']

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=1)

# Set up cross validation validator
criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

random_grid = {'criterion': criterion,
               'splitter': splitter,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

def randomSearchCV(the_grid):
    decision_tree_classifier = DecisionTreeClassifier(random_state=0)
    rtc_random = RandomizedSearchCV(estimator = decision_tree_classifier,
                                    param_distributions = the_grid, n_iter = 100,
                                    cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rtc_random.fit(features_train, target_train)

    dt_class_bland = DecisionTreeClassifier(random_state=0)
    dt_class_bland.fit(features_train, target_train)
    y_predict = dt_class_bland.predict(features_test)
    acc = accuracy_score(target_test, y_predict)
    print("Bland accuracy score:", acc)
    return rtc_random

if __name__ == '__main__':
    rtc = randomSearchCV(random_grid)
    print("Best parameters:", rtc.best_params_)
    print("Best score:", format(rtc.best_score_, '%'))
    print("Error score:", rtc.error_score)
    print("Scoring?", rtc.scoring)
    the_predict = rtc.predict









