#pls add this script to the location having the dataset.
#Usage: python AutoFeatureSelector.py dataset.csv <target_col> 5 pearson,rfe,chi-square,log-reg,ef,lgbm

import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import argparse


def cor_selector(X, y,num_feats):
    cor_list = []
    
    # Calculate the Pearson correlation coefficient between each feature in X and the target variable y
    for i in range(X.shape[1]):
        cor = np.corrcoef(X.iloc[:, i], y)[0, 1]
        cor_list.append(cor)
    
    # Converting correlation list to absolute values
    cor_list = np.abs(cor_list)
    
    # Get the indices of the top num_feats features based on correlation
    top_indices = np.argsort(cor_list)[-num_feats:]
    
    # Initialize a boolean array with the same length as the number of features in X
    cor_support = np.zeros(X.shape[1], dtype=bool)
    
    # Set the top features to True in the boolean array
    cor_support[top_indices] = True
    
    cor_feature = X.columns[cor_support].tolist()
    
    return cor_support, cor_feature


def chi_squared_selector(X, y, num_feats):
    chi_support = []
    chi_feature = []
    chi_selector = SelectKBest(score_func=chi2, k=num_feats)
    chi_selector.fit(X, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.columns[chi_support].tolist()
    # Your code ends here
    return chi_support, chi_feature


def rfe_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(solver='liblinear', random_state=42)

    rfe = RFE(estimator=model, n_features_to_select=num_feats, step=10, verbose=0)
    rfe.fit(X_scaled, y)
    rfe_support = rfe.get_support()
    rfe_feature = X.columns[rfe_support].tolist()
    
    return rfe_support, rfe_feature




def embedded_log_reg_selector(X, y, num_feats):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    model.fit(X_scaled, y)
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    X_selected = selector.transform(X_scaled)
    embedded_lr_support = selector.get_support()
    embedded_lr_feature = list(np.array(X.columns)[embedded_lr_support])
    return embedded_lr_support, embedded_lr_feature




def embedded_rf_selector(X, y, num_feats):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    embedded_rf_support = selector.get_support()
    embedded_rf_feature = list(np.array(X.columns)[embedded_rf_support])
    return embedded_rf_support, embedded_rf_feature



def embedded_lgbm_selector(X, y, num_feats):
    model = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    model.fit(X, y)
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    embedded_lgbm_support = selector.get_support()
    embedded_lgbm_feature = list(np.array(X.columns)[embedded_lgbm_support])
    return embedded_lgbm_support, embedded_lgbm_feature



def preprocess_dataset(dataset_path, y_name):
    df_tmp = pd.read_csv(dataset_path)

    # only keeping the columns where atmost 30% of entries have null values, rest will be deleted. 
    data_tmp = df_tmp.loc[:, df_tmp.isnull().sum() < 0.3*len(df_tmp)].copy()
    
    # dropping rows with null values.
    data = data_tmp.dropna()
    data = pd.DataFrame(data,columns=data.columns)
    try:
    	data = data.drop(columns=['Unnamed: 0'])
    except:
    	pass

    #Splitting features basis their type - categorical/numerical. 	
    numcols = data.select_dtypes(exclude=['object']).columns
    catcols = data.select_dtypes(include=['object']).columns
    
    encoder = LabelEncoder()
    for col in catcols:
        data[col] = encoder.fit_transform(data[col])
    X = data.drop(columns=y_name)
    y = data[y_name]
    return X, y



def autoFeatureSelector(dataset_path, y_name, features_count, methods):
    combined_list = []
    sorted_items = []
    final_list = []
    cor_feature = []
    chi_feature = []
    rfe_feature = []
    embedded_lr_feature = []
    embedded_rf_feature = []
    embedded_lgbm_feature = []

    
    # preprocessing
    print("Starting Preprocessing........")
    X, y = preprocess_dataset(dataset_path, y_name)
    print("Preprocessing done........")
    num_feats = features_count
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        print("Runnning Pearson Correlation.....")
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        print("Runnning Chi-Square.....")
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        print("Runnning RFE.....")
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        print("Runnning Embedded.....")
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        print("Runnning Random Forest.....")
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        print("Runnning Light GBM.....")
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    combined_list = cor_feature + chi_feature + rfe_feature + embedded_lr_feature + embedded_rf_feature + embedded_lgbm_feature
    item_counts = Counter(combined_list)

    # Sort items by their frequency (highest first)
    sorted_items = [item for item, count in item_counts.most_common()]

    # Select the top N items based on user input
    final_list = sorted_items[:features_count] if len(sorted_items) >= features_count else sorted_items
    return final_list



def main():
    parser = argparse.ArgumentParser(description='Feature Selection from CSV.')
    parser.add_argument('file', type=str, help='Path to the CSV file')
    parser.add_argument('target', type=str, help='Target column name')
    parser.add_argument('n_features', type=int, help='Number of features to select')
    parser.add_argument('models', type=str, help='Comma-separated list of models - pearson, chi-square, rfe, log-reg, rf, lgbm')

    args = parser.parse_args()

    # Convert comma-separated models into a list
    model_list = args.models.split(',')

    best_features = autoFeatureSelector(args.file, args.target, args.n_features, model_list)
    print("Top " + str(args.n_features) + " features are: ")
    for feature in best_features:
    	print(feature)

if __name__ == '__main__':
    main()