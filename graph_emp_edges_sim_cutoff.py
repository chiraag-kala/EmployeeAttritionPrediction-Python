"""
Step 0 - Before running this function, please create a cosine similartiy matrix csv
- Use script 'graph_similarity_csv.py' 
"""


def edge_calc(threshold):
  
  import os
  import pandas as pd
  import numpy as np
  import pickle
  import sys
  import re
  os.chdir("/home/cdsw/Data")

  sim_mtx = pd.read_csv('cosine_similarity_v2.csv')

  sim_mtx = sim_mtx.set_index('Employee_ID')

  #Removing Symmetric Values + Diagonal of 1s
  sim_mtx = sim_mtx.mask(np.tril(np.ones(sim_mtx.shape)).astype(np.bool))

  sim_table = sim_mtx[sim_mtx > threshold].stack().reset_index()
  
  sim_table = sim_table[['Employee_ID','level_1']]
  
  sim_table.rename(columns={"Employee_ID":"Emp_1","level_1":"Emp_2"},inplace=True)

  return sim_table

