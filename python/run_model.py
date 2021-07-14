import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

agency_df = pd.read_csv("./data/agency_df_used.csv")
print("Agency df shape:", agency_df.shape)

# Set features and target
target = agency_df['transaction_type']
features = agency_df.loc[:, agency_df.columns != 'transaction_type']

# Create training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.25, random_state=1)

if __name__ == '__main__':
    pkl_filename = './model/XGB_model.pkl'
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    # Calculate the accuracy score and predict target values
    score = pickle_model.score(features_test, target_test)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(features_test)









