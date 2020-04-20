from hclv import create_hclv
hclv = create_hclv()

hclv.columns


#Choose Relevant COlumns for Comparison ----------

df = hclv[['Employee_ID','BUCluster_x','Title_Grouped','Division_x','Function_x', 'Job_Band_x','Manager Employee_ID']]
df.head()

# Convert strings -------------------------------

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df = df.applymap(str)

cols = ['BUCluster_x','Title_Grouped','Division_x','Function_x', 'Job_Band_x','Manager Employee_ID']

for col in cols:
  df[col] = le.fit_transform(df[col])

# Find Combinations of Employees --------------------

import itertools
from itertools import compress, product
from itertools import combinations
import pandas as pd

def rSubset(arr,r):
  return pd.DataFrame(list(combinations(arr,r)))

edges = rSubset(df['Employee_ID'],2)
edges.columns = ['Employee 1', 'Employee 2']


## Join Emp1 ----------------------------------------------------

edges = edges.merge(df, left_on= 'Employee 1', right_on = 'Employee_ID', how = 'left')

edges = edges.rename(columns={"BUCluster_x":"BU 1",
                      "Title_Grouped":"Title 1",
                      "Function_x":"Function 1",
                      "Division_x":"Division 1",
                      "Job_Band_x":"Job Band 1",
                      "Manager Employee_ID": "Manager 1"})
                      
edges = edges.drop(['Employee_ID'], axis = 1)


# Join Emp 2 ------------------------------------------------------
edges = edges.merge(df, left_on='Employee 2', right_on = 'Employee_ID', how = 'left')

edges = edges.rename(columns={"BUCluster_x":"BU 2",
                      "Title_Grouped":"Title 2",
                      "Function_x":"Function 2",
                      "Division_x":"Division 2",
                      "Job_Band_x":"Job Band 2",
                      "Manager Employee_ID": "Manager 2"})

edges = edges.drop(['Employee_ID'], axis = 1)
edges.head()


# Check where ----------------------------------------------------
import numpy as np

edges['Function'] = np.where(edges['Function 1'] == edges['Function 2'],1,0)
edges['BU'] = np.where(edges['BU 1'] == edges['BU 2'],1,0)
edges['Division'] = np.where(edges['Division 1'] == edges['Division 2'],1,0)
edges['Job Band'] = np.where(edges['Job Band 1'] == edges['Job Band 2'],1,0)
edges['Manager'] = np.where(edges['Manager 1'] == edges['Manager 2'],1,0)
edges['Title'] = np.where(edges['Title 1'] == edges['Title 2'],1,0)


# Drop Columns --------------------------------------------------------

edges = edges.drop(['Function 1','Function 2', 
'Job Band 1', 'Job Band 2','BU 1', 'BU 2', 'Title 1', 'Title 2',
'Manager 1', 'Manager 2', 'Division 1', 'Division 2'], axis = 1)

#Output:
edges.head()
edges.info()


edges['weights'] = edges['Function'] + edges['BU'] + edges['Division']+ edges['Job Band'] + edges['Title'] + edges['Manager']
edges.head()

#x.to_csv('emp_edges.csv',index=False)

edges.head()
edges.weights.value_counts()
