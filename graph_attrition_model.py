#Standard Imports 
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import os

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegressionCV

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator, FullBatchNodeGenerator
from stellargraph.layer import GraphSAGE, GCN, GAT
from stellargraph import globalvar

from tensorflow.keras import layers, optimizers, losses, metrics, Model, models
from sklearn import preprocessing, feature_extraction
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#Pre-Defined Functions

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def plot_history(history):
    metrics = sorted(set([remove_prefix(m, "val_") for m in list(history.history.keys())]))
    for m in metrics:
        # summarize history for metric m
        plt.plot(history.history[m])
        plt.plot(history.history['val_' + m])
        plt.title(m, fontsize=18)
        plt.ylabel(m, fontsize=18)
        plt.xlabel('epoch', fontsize=18)
        plt.legend(['train', 'validation'], loc='best')
        plt.show() 

"""
Step 1 - Create Network X Graph with Nodes (Employees) and Edges (Similarity Based on Threshold)

This step uses the following files:
- graph_emp_edges_sim_cutoff.py
- hclv.py'
"""
        
### Import HCLV Dataframe          
cd
from hclv import create_hclv_tb
hclv = create_hclv_tb()


### Import Similarity Thereshold DataFrame 
cd
from graph_emp_edges_sim_cutoff import edge_calc

edges = edge_calc(threshold=0.5) #Set Threshold 

edges = edges[(edges.Emp_1.isin(hclv.Employee_ID)) & (edges.Emp_2.isin(hclv.Employee_ID))]

#Creating Graph and Adding Nodes + Edges
G = nx.Graph()
G.add_nodes_from(hclv.Employee_ID)
G.add_edges_from(list(edges.itertuples(index=False,name=None)))

G.number_of_nodes()
G.number_of_edges()

#Clear Memory
del(edges)


"""
Step 2 - Basic Feature Engineering (Select Columns, Apply Transformations, etc.)
"""
### Create Node Features Table
node_features = hclv[['Churn',
                      'Employee_ID',
                      'Most_Recent_MBO',
                      'Compa Ratio',
                      'Years of Service',
                      'Unique_Level_Changes',
                      'Manager Switches',
                      'Office_Commute_Distance',
                      'Manager_Employee_Count',
                      'Function_Count',
                      'Division_Count',
                      'BU_Count',
                      'Band_Count',
                      'Downward_Level_Counts',
                      'Upward_Level_Counts',
                      'Age_Bucket',
                      'BUCluster_x',
                      'Division_x',
                      'Function_x',
                      'Highest Degree',
                      'Title',
                      'Level_x']]


node_features= shuffle(node_features)

### 1. Missing Values
node_features.Most_Recent_MBO.fillna(node_features.Most_Recent_MBO.median(),inplace=True)
node_features.Office_Commute_Distance.fillna(node_features.Office_Commute_Distance.median(),inplace=True)
node_features['Highest Degree'].fillna(hclv['Highest Degree'].mode()[0],inplace=True)
node_features.Manager_Employee_Count.fillna(0,inplace=True)

### 2. One Hot Encode - Retain Employee ID
emp_IDs = node_features['Employee_ID']
node_features = node_features.drop(['Employee_ID'],axis=1)
node_features = pd.get_dummies(node_features)
print('Shape Before Emp_IDs:',node_features.shape)
node_features = pd.merge(node_features,emp_IDs,left_index=True,right_index=True)
print('Shape After Emp_IDs:',node_features.shape)
node_features = node_features.rename(columns={'Age_Bucket_<25':"Age_Bucket_Under_25"})
node_features = node_features.set_index('Employee_ID')

### 3. Remove Target Column (Churn) from Node Features 
node_features_wc = node_features.drop(columns=['Churn'])


### 4. Apply Power Transformation
from sklearn import preprocessing
pt = preprocessing.PowerTransformer(method='yeo-johnson',standardize=True)
df_values = node_features_wc.iloc[:,:].values
df_values_log = pt.fit_transform(df_values)
node_features_wc.iloc[:,:] = df_values_log
node_features_wc.shape
node_features_wc.head()

"""
Step 3 - Create Stellargraph Object (Graph with Node Features to Train)
- Node Features should not include Target (Churn) 
- Create Target Column 
"""

import stellargraph as sg
Gx = sg.StellarGraph(G,node_features=node_features_wc)

node_features_target = node_features['Churn']

"""
Step 4 - Split Data - Train and Test Sets
"""

### Option 1 - Use ALL Data to Train 
train_data = node_features_wc
train_targets = node_features_target
train_targets = train_targets.values

train_data.shape
train_targets.shape


### Option 2 - Future Holdout Test Set 
os.chdir("/home/cdsw/Data")

sample_ID = pd.read_csv('July1_holdout.csv')
sample_ID.head()
               
test_IDs = list(sample_ID.Employee_ID)

train_data = node_features_wc[~node_features_wc.index.isin(test_IDs)]
train_targets = node_features_target[~node_features_target.index.isin(test_IDs)]
train_targets = train_targets.values
                                              
test_data = node_features_wc[node_features_wc.index.isin(test_IDs)]
test_targets =node_features_target[node_features_target.index.isin(test_IDs)]
test_targets = test_targets.values

test_data.shape
test_targets.shape
train_data.shape
train_targets.shape



### Option 3 - 80% Train / 20% Test

train_data, test_data, train_targets, test_targets = train_test_split(node_features_wc,
                                         node_features_target,
                                         train_size=0.80,
                                         random_state=3212)
train_data.shape
test_data.shape

train_targets.shape
test_targets.shape


"""
Step 5 - Create Class Weights
- Class Weights will be used during training to account for the class imbalance. 
"""

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                     np.unique(train_targets), 
                                     train_targets)

train_class_weights = dict(zip(np.unique(train_targets), 
                               class_weights))
train_class_weights



"""
Step 6 - Create GraphSAGE Generator and Train Model 
"""

### Define Basic Hyperparameters 
batch_size = 50

num_samples = [20, 10] #Number of Nodes to Sample on Each Hop (20 in First Hop, 10 in Second Hop)

epochs = 20


### Load GraphSAGENodeGenerator
generator = GraphSAGENodeGenerator(Gx, batch_size, num_samples)

### Train Model
train_gen = generator.flow(train_data.index, 
                           train_targets, 
                           shuffle=True)

base_model = GraphSAGE(layer_sizes=[32, 32],
                       generator=train_gen,
                       bias=True,
                       dropout=0.5)

x_inp, x_out = base_model.default_model(flatten_output=True)

prediction = layers.Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=prediction)

model.compile(optimizer=optimizers.Adam(lr=0.005),
              loss=losses.binary_crossentropy,
              metrics=["acc"])

history = model.fit_generator(
  train_gen,
  epochs=epochs,
  verbose=2,
  shuffle=False,
  class_weight=train_class_weights,
  use_multiprocessing=True)


"""
Step 7 - Predict Using Model Fitted 
"""

all_nodes = node_features.index

all_gen = generator.flow(all_nodes)


all_predictions = model.predict_generator(all_gen).squeeze()[..., np.newaxis]

all_predictions.shape

all_predictions_df = pd.DataFrame(all_predictions, 
                                  index=node_features.index)



"""
Step 8 - Model Validation - All Data and Test Set 
"""

### View Performance on All Data 

all_predictions_class = ((all_predictions>0.5)*1).flatten()

all_df = pd.DataFrame({"Predicted_score": all_predictions.flatten(), 
                        "Predicted_class": all_predictions_class, 
                        "True": node_features_target})


pd.crosstab(all_df['True'], all_df['Predicted_class'])


### View Performance on Test Set

test_preds = all_predictions_df.loc[test_data.index, :]

test_preds.shape

test_predictions = test_preds.values

test_predictions_class = ((test_predictions>0.5)*1).flatten()
test_df = pd.DataFrame({"Predicted_score": test_predictions.flatten(), 
                        "Predicted_class": test_predictions_class, 
                        "True": test_targets})


### Test Set Performance Metrics 

#ROC
roc_auc = metrics.roc_auc_score(test_df['True'].values, 
                                test_df['Predicted_score'].values)
print("The AUC on test set:\n")
print(roc_auc)

#Confusion Matrix
pd.crosstab(test_df['True'], test_df['Predicted_class'])

#Classification Report
print(classification_report(test_df['True'],test_df['Predicted_class']))

#TP, FP, TN, and FN
from sklearn.metrics import classification_report
CM = confusion_matrix(test_df['True'],test_df['Predicted_class'])

TN = CM[0][0]/len(test_df['True'])
FN = CM[1][0]/len(test_df['True'])
TP = CM[1][1]/len(test_df['True'])
FP = CM[0][1]/len(test_df['True'])

print('TN:',TN)
print('FN:',FN)
print('TP:',TP)
print('FP:',FP)

