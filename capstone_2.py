# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 16:18:39 2017

@author: amybrown
"""

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

churn_data = pd.read_csv('/Users/amybrown/Thinkful/Capstone/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn_data['SeniorCitizen'] = churn_data['SeniorCitizen'].astype(str)

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
                           
churn_dummies['No. Products'] = churn_dummies['Security:Yes'] + churn_dummies['Backup:Yes'] \
+ churn_dummies['Protection:Yes'] + churn_dummies['Support:Yes'] \
+ churn_dummies['StreamTV:Yes'] + churn_dummies['StreamMov:Yes'] 

#%%
""" prepare data for modeling """

# target data
churn_outcome = churn_dummies['Churn:Yes']
y = np.where(churn_outcome == 1, 1, 0)
#y = y.reshape(7043,1)
#y = churn_outcome

churn_features = churn_dummies[['Female', 'SeniorCitizen', 'Partner:Yes', 'Dependent:Yes', 'tenure',
                                'Phone:Yes', 'Multi-lines:Yes', 'Multi-lines:No', 
                                'Internet:DSL', 'Internet:Fiber optic', 'No. Products', 
                                'Contract:One year', 'Contract:Two year', 'PaperlessBill:Yes', 
                                'Method:Mailed check', 'Method:Bank transfer (automatic)', 
                                'Method:Credit card (automatic)']]
X = churn_features
      
X = churn_features.as_matrix().astype(np.float)
scale = StandardScaler()
X = scale.fit_transform(X)

#%%
""" 0 rate classifier """

y_true = y

all_churn_df = churn_dummies
all_churn_df['all_churn'] = 1.0
all_churn = all_churn_df['all_churn']
y_pred = np.where(all_churn == 1, 1, 0)


#%%""" Models """

from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

classifiers = (LR,DT,RF)
score_metric = 'accuracy'

def optimization(classifier):
    if classifier == LR:
        param_grid = {'class_weight': ['balanced'], 'solver': ['liblinear', 'sag'], 'cv': [5], 'refit': ['True', 'False']}
    if classifier == DT:
        param_grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'class_weight': ['balanced']}
    if classifier == RF:
        param_grid = {'n_estimators': [10,20,30], 'criterion': ['gini', 'entropy'], 'bootstrap': ['True']}
    print(str(classifier))
    print('Number of tested models: %i' % np.prod([len(param_grid[element]) for element in param_grid]))
    search = GridSearchCV(classifier(), param_grid, cv=10)
    search.fit(X,y)
    print('Best parameters: %s' % search.best_params_)
    print('Best score: ' + str(search.best_score_))


lr_params = {'refit': 'True', 'solver': 'sag', 'class_weight': 'balanced', 'cv': 5}
dt_params = {'splitter': 'best', 'criterion': 'entropy', 'class_weight': 'balanced'}
rf_params = {'n_estimators': 10, 'criterion': 'entropy', 'bootstrap': 'True'}


#%%
""" cross validation """

def run_cv(X,y,clf_class,**kwargs):
    kf = KFold(len(y),n_folds=5,shuffle=True, random_state=42)
    y_pred = y.copy()

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred
    
def avg_correct(y_true, y_pred):
    return np.mean(y_true == y_pred)
   
#%%

print("Zero Rate:")
print("%.3f" % avg_correct(y_true, y_pred))

print('Logistic Regression:')
print("%.3f" % avg_correct(y, run_cv(X,y,LR,refit=True, solver='sag', class_weight='balanced',cv=5)))

print('Decision Tree:')
print("%.3f" % avg_correct(y, run_cv(X,y,DT,splitter='best',criterion='entropy',class_weight='balanced')))

print('Random Forest:')
print("%.3f" % avg_correct(y, run_cv(X,y,RF,n_estimators=10, criterion='entropy',bootstrap=True)))

#%% Confusion Matrices
y = np.array(y)
class_names = np.unique(y)

#confusion_matrices = [
#    ( "Logistic Regression", confusion_matrix(y,run_cv(X,y,LR)) ),
#    ( "Decision Tree", confusion_matrix(y,run_cv(X,y,DT)) ),
#    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),
#]

zr_cm = confusion_matrix(y_true, y_pred)
lr_cm = confusion_matrix(y,run_cv(X,y,LR))
dt_cm = confusion_matrix(y,run_cv(X,y,DT))
rf_cm = confusion_matrix(y,run_cv(X,y,DT))

#%%

def spec_measure(confusion_matrix):
    false_negs = confusion_matrix[0,1]
    true_negs = confusion_matrix[1,1]
    total_negs = false_negs + true_negs
    specificity = false_negs/total_negs
    print(specificity)

def false_neg(confusion_matrix):
    true_pos = confusion_matrix[0,0]
    false_pos = confusion_matrix[1,0]
    total_pos = true_pos + false_pos
    sensitivity = true_pos/total_pos
    false_negative_rate = 1 - sensitivity
    print(sensitivity)
    
def false_neg2(confusion_matrix):
    false_negs = confusion_matrix[0,1]
    true_pos = confusion_matrix[0,0]
    denom = false_negs + true_pos
    false_negative_rate = false_negs/denom
    print(false_negative_rate)

