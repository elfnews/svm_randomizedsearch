# Import our libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC

from spam_ensembles import print_metrics

sns.set(style="ticks")

# Read in our dataset
diabetes = pd.read_csv('diabetes_case_study_data.csv')

# Take a look at the first few rows of the dataset
print(diabetes.head())
print(diabetes.describe())

sns.pairplot(diabetes, hue="Outcome")
sns.heatmap(diabetes.corr(), annot=True, cmap="YlGnBu")

y = diabetes['Outcome']
X = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# build a classifier
clf_rf = RandomForestClassifier()
 
# Set up the hyperparameter search
param_dist = {"max_depth": [3, None],
              "n_estimators": list(range(10, 200)),
              "max_features": list(range(1, X_test.shape[1] + 1)),
              "min_samples_split": list(range(2, 11)),
              "min_samples_leaf": list(range(1, 11)),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fist teh model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
rf_preds = random_search.best_estimator_.predict(X_test)

print_metrics(y_test, rf_preds, 'random forest')

# build a classifier for a ada boost
clf_ada = AdaBoostClassifier()

# Set up the hyperparameter search
# Look at setting up your search for n_estimators, learning_rate
param_dist = {"n_estimators": [10, 100, 200, 400],
              "learning_rate": [0.001,
                                0.005,
                                0.01,
                                0.05,
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                1,
                                2,
                                10,
                                20]}

# Run a randomized search over the hyperparameters
ada_search = RandomizedSearchCV(clf_ada, param_distributions=param_dist)

# Fit the model on the training data
ada_search.fit(X_train, y_train)

# Make predictions on the test data
ada_preds = ada_search.best_estimator_.predict(X_test)

print_metrics(y_test, ada_preds, 'adaboost')

# build a classifier for support vector machines
clf_svc = SVC()

# Set up the hyperparameter search
# Look at setting up your search for C (recoomend 0-10 range),
# kernel, and degree
param_dist = {"C": [0.1, 0.5, 1, 3, 5],
              "kernel": ['linear', 'rbf']}

# Run a randomized search over the hyperparameters
svc_search = RandomizedSearchCV(clf_svc, param_distributions=param_dist)

# Fit the model on the training data
svc_search.fit(X_train, y_train)

# Make predictions on the test data
svc_preds = svc_search.best_estimator_.predict(X_test)

print_metrics(y_test, svc_preds, 'svc')

features = diabetes.columns[:diabetes.shape[1]]
importances = random_search.best_estimator_.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

plt.show()
