#Standard Imports
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,make_scorer
from xgboost import plot_importance
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

import ml_timemodel
ml_frame_tb = ml_timemodel.main()
ml_frame_tb = shuffle(ml_frame_tb)

""""
Step 1 - Create X_Train | X_Test Sets 
"""

#Select 272 Individuals who left after June 30, 2019 + 272 Random Active Employees 
test_set_idx = shuffle(list(ml_frame_tb[ml_frame_tb['Termination Date'] > '2019-06-30'].index)
                       +list(ml_frame_tb[ml_frame_tb['Termination Date'].isnull()].sample(272,random_state=21323).index))

X_test = shuffle(ml_frame_tb[ml_frame_tb.index.isin(test_set_idx)])

X_train = shuffle(ml_frame_tb[~ml_frame_tb.index.isin(test_set_idx)])


"""
Step 2 - Subset for Churn Only and Drop Other Columns
"""
X_train = X_train[X_train['Churn'] == 1]
y_train_DOC = X_train['Days_to_Churn']
y_train_DOC= pd.to_numeric(y_train_DOC.dt.days,downcast='integer')
X_train = X_train.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)

X_test = X_test[X_test['Churn'] == 1]
y_test_DOC = X_test['Days_to_Churn']
y_test_DOC= pd.to_numeric(y_test_DOC.dt.days,downcast='integer')
X_test = X_test.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)

print(X_train.shape)
print(X_test.shape)
print(y_test_DOC.shape)
print(y_train_DOC.shape)

                                         
""""                                                  
Step 3. Model Fit and Predict 
""" 
XGBR = XGBRegressor()

XGBR.fit(X_train,y_train_DOC)

#Train Prediction
train_pred = XGBR.predict(X_train)
print('Train RMSE:',np.sqrt(mean_squared_error(y_train_DOC,train_pred)))

#Test Prediction
pred = XGBR.predict(X_test)
print('Test RMSE:',np.sqrt(mean_squared_error(y_test_DOC,pred)))


""" 
Step 4. Feature Importances 
"""
import shap

shap.initjs()
explainer = shap.TreeExplainer(XGBR)
shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value,shap_values[0,:],X_train.iloc[0,:])

shap.summary_plot(shap_values,X_train)
shap.summary_plot(shap_values,X_train,plot_type='bar')
          