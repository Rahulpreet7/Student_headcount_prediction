import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle
import sys


file = "merged_data.csv"
data = pd.read_csv(file)

y = data[['Basement','Main_floor', "Second_floor","Third_floor","Fourth_floor"]]
X= data[['Day','Month','Year','Time']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42,stratify = data["Time"])

rf_reg = RandomForestRegressor(random_state =42,max_features=2, n_estimators=30)
rf_reg.fit(X_train, y_train)
prediction_rf = rf_reg.predict(X_test)

# Saving model to disk
pickle.dump(rf_reg, open('model.pkl','wb'))