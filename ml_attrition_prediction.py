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

"""
Step 1 -Import preprocessed, ML ready dataframe from the script 'ml_timemodel.py'
Uses the following scripts as dependencies
- ml_timemodel.py
- hclv.py 
"""

#Import DataFrame
cd
import ml_timemodel
ml_frame_tb = ml_timemodel.main()
ml_frame_tb = shuffle(ml_frame_tb)

#Checks on the Frame
ml_frame_tb.head()
ml_frame_tb.shape
ml_frame_tb.columns[ml_frame_tb.isnull().any()].tolist()


""""
#Step 2 - Train/Test Divide - Here we used two different train/test split strategies: 
- 80%/20% Split Strategy - Randomly Selected 80/20 Split 
- Future Holdout Strategy - Created A Set For Attrition Starting June 2019 
"""

### 1. 80%/20% Split Strategy

X = ml_frame_tb.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)
y = ml_frame_tb['Churn']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.20,
                                                    random_state=20112,
                                                    stratify=y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#### 2. Future Holdout Strategy

#Select 272 Individuals who left after June 30, 2019 + 272 Random Active Employees 
test_set_idx = shuffle(list(ml_frame_tb[ml_frame_tb['Termination Date'] > '2019-06-30'].index)
                       +list(ml_frame_tb[ml_frame_tb['Termination Date'].isnull()].sample(272,random_state=21323).index))

X_test = shuffle(ml_frame_tb[ml_frame_tb.index.isin(test_set_idx)])

X_train = shuffle(ml_frame_tb[~ml_frame_tb.index.isin(test_set_idx)])

#Drop Relevant Columns 

y_train = X_train['Churn']
y_train_DOC = X_train['Days_to_Churn']
X_train = X_train.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)

y_test = X_test['Churn']
y_test_DOC = X_test['Days_to_Churn']
X_test = X_test.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


""""
#Step 3 - Random Forest Model - Fit and Predict
"""

#Fit the Model
rf = RandomForestClassifier(n_estimators=125,
                            max_depth=8,
                            max_features=50,
                            min_samples_split=10,
                            class_weight='balanced',
                            random_state = 555,
                            bootstrap = True)


rf.fit(X_train,y_train)


#Train Prediction
train_pred = rf.predict(X_train)
print(classification_report(y_train,train_pred))
pd.crosstab(y_train,train_pred)


#Test Prediction
pred = rf.predict(X_test)
print(classification_report(y_test,pred))
pd.crosstab(y_test,pred)


#Calculation for TP, FP, FN and FP - Test Set Only
CM = confusion_matrix(y_test,pred)

TN = CM[0][0]/len(y_test)
FN = CM[1][0]/len(y_test)
TP = CM[1][1]/len(y_test)
FP = CM[0][1]/len(y_test)

print('TN:',TN)
print('FN:',FN)
print('TP:',TP)
print('FP:',FP)


""""
#Step 4 - Cross Validation and Feature Importances
"""

### Cross Validation with 5 Folds on Full Original Data

X = ml_frame_tb.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)
y = ml_frame_tb['Churn']

originalclass = []
predictecdlcass = []

def classification_report_w_accuracy(y_true,y_pred):
  originalclass.extend(y_true)
  predictecdlcass.extend(y_pred)
  print(classification_report(y_true,y_pred))
  return accuracy_score(y_true,y_pred)

rf = RandomForestClassifier(n_estimators=125,
                            max_depth=8,
                            max_features=50,
                            min_samples_split=10,
                            class_weight='balanced',
                            random_state = 555,
                            bootstrap = True)

  
cross_validate(rf,X,y,cv=5,scoring=make_scorer(classification_report_w_accuracy))

### Feature Importance on Full Original Data 

#Initializing Model for Feature Importance 

X = ml_frame_tb.drop(['Churn','Days_to_Churn','Termination Date','Employee_ID'],axis=1)
y = ml_frame_tb['Churn']

rf = RandomForestClassifier(n_estimators=125,
                            max_depth=8,
                            max_features=50,
                            min_samples_split=10,
                            class_weight='balanced',
                            random_state = 555,
                            bootstrap = True)



rf.fit(X,y)

#Using SHAP for Feature Importances

import shap
shap.initjs()

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values,X)
shap.summary_plot(shap_values[1],X)


""""
Other: Oversampling/Undersampling Code
"""

#Oversampling Majority Class - Attrition
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=32,sampling_strategy=0.90)

X_train.shape
y_train.shape
y_train.sum()
y_train.sum()/len(y_train)

X_train_res, y_train_res = ada.fit_sample(X_train, y_train.ravel())

print(X_train_res.shape)
print(y_train_res.shape)
print(y_train_res.sum()/len(y_train_res))

#Undersampling Minority Class - Active Employees
from imblearn.under_sampling import RandomUnderSampler

X_train.shape
y_train.shape
y_train.sum()
y_train.sum()/len(y_train)

U_sampler = RandomUnderSampler(random_state=43434,sampling_strategy=0.8)

X_train_Ures, y_train_Ures = U_sampler.fit_sample(X_train, y_train.ravel())
X_train_Ures.shape
y_train_Ures.shape
y_train_Ures.sum()/len(y_train_Ures)


""""
Other: Check for Approriate Threshold
- Dependency - Only Valid for Future Hold Out Testing Strategy (Run Step 2 - Part 2)
"""

rf.fit(X_train,y_train)

pred_prob = rf.predict_proba(X_test)
pred_prob = pred_prob[:,1]


thereshold_list = []
accuracy_list = []
recall_attr_list = []
recall_active_list = []
precision_attr_list = []


for i in range(20,80,1):
  
  print(i/100)
  
  pred_new = np.where(pred_prob >= i/100,1,0)
  
  x = pd.crosstab(y_test,pred_new)
  
  acc = (x.iloc[0,0] + x.iloc[1,1])/(544)
  
  recall_atr = (x.iloc[1,1])/(272)
  
  recall_actv = (x.iloc[0,0])/(272)
  
  precision_atr = x.iloc[1,1]/((x.iloc[1,1])+(x.iloc[0,1]))
  
  #print(acc)
  #print(recall_atr)
  
  thereshold_list.append(i/100)
  accuracy_list.append(acc)
  recall_attr_list.append(recall_atr)
  recall_active_list.append(recall_actv)
  precision_attr_list.append(precision_atr)
    

final_perf = pd.DataFrame({'Probability Cutoff':thereshold_list,
    'Overall Accuracy':accuracy_list,
    'Recall for Attrition':recall_attr_list,
    'Recall for Active Employees':recall_active_list,
    'Precision for Attrition':precision_attr_list})


sns.lineplot(x='Probability Cutoff',y='Overall Accuracy',data=final_perf)
sns.lineplot(x='Probability Cutoff',y='Recall for Attrition',data=final_perf)
sns.lineplot(x='Probability Cutoff',y='Recall for Active Employees',data=final_perf)
sns.lineplot(x='Probability Cutoff',y='Precision for Attrition',data=final_perf)


""""
Other: Model Interpretation - Code for Partial Dependence Plots, etc. 
"""

### PDP Box Sample Code
from pdpbox import pdp, info_plots

fig, axes, summary_df = info_plots.target_plot(df=ml_frame_tb,
                                               feature='Most_Recent_MBO',
                                               feature_name='MBO',
                                               target='Churn',
                                               show_percentile=True)


fig, axes, summary_df = info_plots.target_plot(df=ml_frame_tb,
                                               feature='Compa Ratio',
                                               feature_name='Compa Ratio',
                                               target='Churn',
                                               show_percentile=True)

X_test['Pred'] = pred
X_test['Actual'] = y_test
X_test_ftlist = (set(X_test.columns) - set(['Pred','Actual']))

pdp_MBO = pdp.pdp_isolate(model=rf,
                         dataset=X_test,
                         model_features=X_test_ftlist,
                         feature='Most_Recent_MBO')  

fig, axes = pdp.pdp_plot(pdp_MBO,'Most_Recent_MBO',plot_pts_dist=True,
                        frac_to_plot=0.9,plot_lines=True,x_quantile=True,
                        show_percentile=True)


fig, axes, summary_df = info_plots.target_plot_interact(df=X_test,
                                                       features=['Most_Recent_MBO','Compa Ratio'],
                                                       feature_names=['MBO','Comp Ratio'],
                                                       target='Pred')

iternl = pdp.pdp_interact(model=rf,
                          dataset=X_test,
                          model_features=X_test_ftlist,
                          features=['Most_Recent_MBO','Compa Ratio'])

fig, axes = pdp.pdp_interact_plot(pdp_interact_out=iternl,
                                  feature_names=['MBO','Comp Ratio'],
                                  plot_type='grid',
                                  x_quantile=True,
                                  plot_pdp=True)


""""
Other: Visualize Random Forest / Decision Tree
"""
### Visualize Random Tree From Random Forest Model 

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import graphviz

#Visualizing Tree #100
estimator = rf.estimators_[100]

dot_data = export_graphviz(estimator,out_file=None,
                feature_names=X_train.columns)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())                                                                                             