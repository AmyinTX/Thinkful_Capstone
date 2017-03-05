# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:05:46 2017

@author: abrown09
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

churn_data = pd.read_csv('/Users/amybrown/Thinkful/Capstone/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data exploration

churn_data.shape # check dimensions
churn_data.dtypes
print(churn_data.head(10)) 
# not sure what tenure vaar means-it must be the amount of time customer has been with company
print(churn_data.describe()) # looks like numerical data is complete. senior citizen is not an age variable but a binary categorical variable
# mean tenure is 32...i'm going to guess this is months and not years because that seems crazy

categorical = churn_data.dtypes[churn_data.dtypes == 'object'].index
print(categorical)

churn_data[categorical].describe() # all data appear complete
# I think total charges needs to be changed to float

churn_data.hist(column='tenure', figsize=(9,6))
churn_data.hist(column='MonthlyCharges', figsize=(9,6))

scatter_matrix(churn_data, alpha=0.2, figsize=(6, 6), diagonal='kde')

# should Yes and No responses be changed to 1s and 0s?
