# ML-AutoFeatureSelector

A simple python utility to select top n features that have highest impact on the target column (y) which we need to predict.
Uses the following methods:
1. Pearson Correlation
2. Chi-Square
3. Recursive Feature Elimination
4. Embedded Logistic Regression
5. Embedded Random Forest
6. Embedded LightGBM

Usage:
1. python3 AutoFeatureSelector.py dataset.csv <target_col> <number_of_features> pearson,rfe,chi-square,log-reg,ef,lgbm
2. python3 AutoFeatureSelector.py dataset.csv price 4 pearson,rfe,log-reg,lgbm
3. python3 AutoFeatureSelector.py dataset.csv label 2 pearson,lgbm
