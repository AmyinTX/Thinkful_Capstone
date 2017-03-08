# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:05:46 2017

@author: abrown09
"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sb
import numpy as np

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

churn_data = pd.read_csv('/Users/amybrown/Thinkful/Capstone/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data exploration

churn_data.shape # check dimensions
churn_data.dtypes

# converting to float isnt working because there are some blank values in the dataframe
# for totalCharges. these are all new customers. can either fill with 0 or fill with
# whatever their monthly charges are.

for val in churn_data['TotalCharges']:
    if churn_data['TotalCharges'] == ' ':
        churn_data['TotalCharges'] == churn_data['MonthlyCharges']
        
#churn_data['TotalCharges'] = churn_data['TotalCharges'].fillna  # doesnt work
 

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
churn_data.hist(column='TotalCharges', figsize=(9,6))

scatter_matrix(churn_data, alpha=0.2, figsize=(6, 6), diagonal='kde')

# should Yes and No responses be changed to 1s and 0s?

# get sense of unique values
for col in churn_data:
    print(col)
    print(churn_data[col].unique())

# list that will need to be dummied:
# gender, partner, dependents, phone service, multiple lines, internet service, online security, online backup, device protection, 
#tech support, streaming tv, 
# streaming_movies, contract, paperless billing, payment method

df_sex = pd.get_dummies(churn_data['gender'])
df_partner = pd.get_dummies(churn_data['Partner'], prefix='Partner',prefix_sep=':')
df_depend = pd.get_dummies(churn_data['Dependents'], prefix='Dependent',prefix_sep=':')
df_phone = pd.get_dummies(churn_data['PhoneService'], prefix='Phone',prefix_sep=':')
df_lines = pd.get_dummies(churn_data['MultipleLines'], prefix='Multi-lines',prefix_sep=':')
df_internet = pd.get_dummies(churn_data['InternetService'], prefix='Internet', prefix_sep=':')
df_secure = pd.get_dummies(churn_data['OnlineSecurity'], prefix='Security', prefix_sep=':')
df_backup =  pd.get_dummies(churn_data['OnlineBackup'], prefix='Backup', prefix_sep=':')
df_protect = pd.get_dummies(churn_data['DeviceProtection'], prefix='Protection', prefix_sep=':')
df_support = pd.get_dummies(churn_data['TechSupport'], prefix='Support', prefix_sep=':')
df_streamtv = pd.get_dummies(churn_data['StreamingTV'], prefix='StreamTV', prefix_sep=':')
df_streammov = pd.get_dummies(churn_data['StreamingMovies'], prefix='StreamMov', prefix_sep=':')
df_contract = pd.get_dummies(churn_data['Contract'], prefix='Contract', prefix_sep=':')
df_billing = pd.get_dummies(churn_data['PaperlessBilling'], prefix='PaperlessBill', prefix_sep=':')
df_payment = pd.get_dummies(churn_data['PaymentMethod'], prefix='Method', prefix_sep=':')
df_churn = pd.get_dummies(churn_data['Churn'], prefix='Churn',prefix_sep=':')



churn_dummies = pd.concat([churn_data, df_sex, df_partner, df_depend, df_phone, df_lines, df_internet,
                           df_secure, df_backup, df_protect, df_support, df_streamtv, df_streammov, df_contract,
                           df_billing, df_payment, df_churn], axis=1)

#histograms
churn_data.hist()
plt.show
churn_dummies.hist()
plt.show()

#density plots
churn_data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
churn_dummies.plot(kind='density', subplots=True,sharex=False)

#boxplots
churn_data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
churn_dummies.plot(kind='box', subplots=True, layout=(4, 11), sharex=False, sharey=False)

# feature to class: distributions of each feature by class
churn_data.groupby('Churn').hist()
churn_dummies.groupby('Churn').hist()

# feature to feature
from pandas.tools.plotting import scatter_matrix
scatter_matrix(churn_data, alpha=0.2, figsize=(6, 6), diagonal='kde')
scatter_matrix(churn_dummies, alpha=0.2, figsize=(6, 6), diagonal='kde')

# plots with seaborne--feature to feature
sb.barplot(x='gender',y='tenure', data=churn_data)
sb.barplot(x='gender', y='MonthlyCharges',data=churn_data)
sb.barplot(x='gender', y='TotalCharges', data=churn_data) 

sb.barplot(x='SeniorCitizen', y='tenure', data=churn_data)
sb.barplot(x='SeniorCitizen', y='MonthlyCharges', data=churn_data)
sb.barplot(x='SeniorCitizen', y='TotalCharges', data=churn_data) #this part doesn't work for some reason

sb.barplot(x='Partner', y='tenure', data=churn_data)
sb.barplot(x='Partner', y='MonthlyCharges', data=churn_data)
sb.barplot(x='Partner', y='TotalCharges', data=churn_data)

sb.barplot(x='Dependents', y='tenure', data=churn_data)
sb.barplot(x='Dependents', y='MonthlyCharges', data=churn_data)
sb.barplot(x='Dependents', y='TotalCharges', data=churn_data)

sb.barplot(x='PhoneService', y='tenure', data=churn_data)
sb.barplot(x='PhoneService', y='MonthlyCharges', data=churn_data)
sb.barplot(x='PhoneService', y='TotalCharges', data=churn_data)

sb.barplot(x='MultipleLines', y='tenure', data=churn_data)
sb.barplot(x='MultipleLines', y='MonthlyCharges', data=churn_data)
sb.barplot(x='MultipleLines', y='TotalCharges', data=churn_data)

sb.barplot(x='InternetService', y='tenure', data=churn_data)
sb.barplot(x='InternetService', y='MonthlyCharges', data=churn_data)
sb.barplot(x='InternetService', y='TotalCharges', data=churn_data)

sb.barplot(x='OnlineSecurity', y='tenure', data=churn_data)
sb.barplot(x='OnlineSecurity', y='MonthlyCharges', data=churn_data)
sb.barplot(x='OnlineSecurity', y='TotalCharges', data=churn_data)

sb.barplot(x='OnlineBackup', y='tenure', data=churn_data)
sb.barplot(x='OnlineBackup', y='MonthlyCharges', data=churn_data)
sb.barplot(x='OnlineBackup', y='TotalCharges', data=churn_data)

sb.barplot(x='DeviceProtection', y='tenure', data=churn_data)
sb.barplot(x='DeviceProtection', y='MonthlyCharges', data=churn_data)
sb.barplot(x='DeviceProtection', y='TotalCharges', data=churn_data)

sb.barplot(x='TechSupport', y='tenure', data=churn_data)
sb.barplot(x='TechSupport', y='MonthlyCharges', data=churn_data)
sb.barplot(x='TechSupport', y='TotalCharges', data=churn_data)

sb.barplot(x='StreamingTV', y='tenure', data=churn_data)
sb.barplot(x='StreamingTV', y='MonthlyCharges', data=churn_data)
sb.barplot(x='StreamingTV', y='TotalCharges', data=churn_data)

sb.barplot(x='StreamingMovies', y='tenure', data=churn_data)
sb.barplot(x='StreamingMovies', y='MonthlyCharges', data=churn_data)
sb.barplot(x='StreamingMovies', y='TotalCharges', data=churn_data)

sb.barplot(x='Contract', y='tenure', data=churn_data)
sb.barplot(x='Contract', y='MonthlyCharges', data=churn_data)
sb.barplot(x='Contract', y='TotalCharges', data=churn_data)

sb.barplot(x='PaperlessBilling', y='tenure', data=churn_data)
sb.barplot(x='PaperlessBilling', y='MonthlyCharges', data=churn_data)
sb.barplot(x='PaperlessBilling', y='TotalCharges', data=churn_data)

sb.barplot(x='PaymentMethod', y='tenure', data=churn_data)
sb.barplot(x='PaymentMethod', y='MonthlyCharges', data=churn_data)
sb.barplot(x='PaymentMethod', y='TotalCharges', data=churn_data)

# seaborn plots--feature to class
sb.barplot(x='Churn',y='tenure', data=churn_data)
sb.barplot(x='Churn', y='MonthlyCharges',data=churn_data)
sb.barplot(x='Churn', y='TotalCharges', data=churn_data) 

sb.countplot(x='Churn', hue='gender', data=churn_data)
sb.countplot(x='Churn', hue='SeniorCitizen', data=churn_data)
sb.countplot(x='Churn', hue='Partner', data=churn_data)
sb.countplot(x='Churn', hue='Dependents', data=churn_data)
sb.countplot(x='Churn', hue='PhoneService', data=churn_data)
sb.countplot(x='Churn', hue='MultipleLines', data=churn_data)
sb.countplot(x='Churn', hue='InternetService', data=churn_data)
sb.countplot(x='Churn', hue='OnlineSecurity', data=churn_data)
sb.countplot(x='Churn', hue='OnlineBackup', data=churn_data)
sb.countplot(x='Churn', hue='DeviceProtection', data=churn_data)
sb.countplot(x='Churn', hue='TechSupport', data=churn_data)
sb.countplot(x='Churn', hue='StreamingTV', data=churn_data)
sb.countplot(x='Churn', hue='StreamingMovies', data=churn_data)
sb.countplot(x='Churn', hue='Contract', data=churn_data)
sb.countplot(x='Churn', hue='PaperlessBilling', data=churn_data)
sb.countplot(x='Churn', hue='PaymentMethod', data=churn_data)
