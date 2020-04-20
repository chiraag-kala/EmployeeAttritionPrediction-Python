'''This version excludes all MBO or OPR scores, as they cannot work with the 
time bucketing method. Once Monu adds the imputation codes '''

import pandas as pd
import numpy as np
import pickle
import sys
import re

pd.set_option('display.max_columns',None)

# --- Cleaning Functions
def create_grouped_columns(hclv):
  """This function creates custom groupings for any high-dimensional columns."""
  
  #'Work Address - City' --> Work_City_Grouped 
  cities_to_group = list(hclv['Work Address - City'].value_counts()[hclv['Work Address - City'].value_counts()<=10].index)
  hclv['Work_City_Grouped'] = hclv['Work Address - City']
  hclv['Work_City_Grouped'][hclv['Work_City_Grouped'].isin(cities_to_group)] = 'Other'
  hclv['Work_City_Grouped'].value_counts()
  
  return hclv

#-------------------------------------------------------------------------------------------

def drop_columns(hclv):
  """This function drops columns that won't be used for the CB Model."""
  
  all_columns = list(hclv.columns)
  
  columns_to_remove = ['Report Date',
                       'Manager Employee_ID',
                       'Position_ID_x',
                       'Manager Position_ID',
                       'Date Filled',
                       'Company Service Date',
                       'Earliest Hire Date',
                       'Original Hire Date_x',
                       'Effective Date for Current Position_x',
                       'Hire Date_x',
                       'Time in Position_x',
                       'FTE',
                       'FTE %',
                       'Time in Job Profile Start Date',
                       'Time Type_x',
                       'Leave Type',
                       'MBO_Team_18',
                       'MBO_Team_17',
                       'MBO_Self_18',
                       'MBO_Self_17',
                      '9-Box 17',
                      '9-Box 18',
                      'MBO_17_Combined',
                      'MBO_18_Combined',
                      'Most_Recent_9_Box']

  high_dim_cols = ['Work Address - City',
                   'Primary Work Address',
                   'Location_x']

  leaver_cols = ['Country',
                 'Business_Title',
                 'Employee Type',
                 'Job Category',
                 'Global Comp Ratio',
                 'Termination Date',
                 'Term Reason',
                 'Length of Service',
                 'Termination Month',
                 'Termination Year']

  historical_append_cols = ['Level_x',
                            'Function_x',
                            'Business Unit_x',
                            'Division_x',
                            'Job_Band_x',
                            'Title',
                            'Title_Grouped']

  final_columns = set(all_columns) - set(high_dim_cols) - set(columns_to_remove) - set(leaver_cols) - set(historical_append_cols)
  
  return hclv[final_columns]

#-----------------------------------------------------------------------------------------------------

def missing_values(hclv):
  """This function imputes any missing values."""
  #BUCluster
  hclv['BUCluster_x'].fillna(hclv['BUCluster_x'].mode()[0],inplace=True)
  hclv['NewBusinessUnit_x'].fillna(hclv['NewBusinessUnit_x'].mode()[0],inplace=True)
  
  #Demographic Info
  hclv['Gender'].fillna(hclv.Gender.mode(),inplace=True)
  hclv['Highest Degree'] = hclv.groupby('BUCluster_x')['Highest Degree'].transform(lambda x: x.fillna(x.mode()[0]))
  hclv['Ethnicity'].fillna(hclv['Ethnicity'].mode()[0],inplace=True)
  hclv['Work_City_Grouped'].fillna("Other",inplace=True)
  
  #Office_Commute_Distance
  hclv['Office_Commute_Distance'] = hclv.groupby('BUCluster_x')['Office_Commute_Distance'].transform(lambda x: x.fillna(x.median()))
  #This one person has his own BUCluster, and has NaN value
  hclv['Office_Commute_Distance'].fillna(hclv['Office_Commute_Distance'].median(),inplace=True)
  
  #MBO Score
  hclv['Most_Recent_MBO'] = hclv.groupby('BUCluster_x')['Most_Recent_MBO'].transform(lambda x: x.fillna(x.median()))
  #There are 16 people whose whole BUCluster misses MBO Score
  hclv['Most_Recent_MBO'].fillna(hclv['Most_Recent_MBO'].median(),inplace=True)
  
  #Merit
  hclv.fillna(0,inplace=True)
  
  
  return hclv

#------------------------------------------------------------------------------------------------------

def numerical_to_bins(hclv):
  
  ### YEARS OF SERVICE
  bins = [0,1,2,5,10,50]
  labels = ['First year','1-2 years','2-5 years','5-10 years','10+ years']
  hclv['YOS_Binned'] = pd.cut(hclv['Years of Service'], bins=bins, labels=labels, include_lowest=True).astype("object")

  ### MBO SCORES
  #hclv['MBO_17_Combined'].quantile([0.10,0.90])
  #bins_17 = [0,50.5,100,124]
  #labels_17 = ['17_bottom_10','17_middle_80','17_top_10']
  #hclv['17_Binned'] = pd.cut(hclv['MBO_17_Combined'], bins=bins_17, labels=labels_17, include_lowest=True).astype("object")

  ### Compa Ratio
  #hclv['Compa Ratio'].quantile([0.10,0.90])
  #bins_comp = [0,0.7420,1.0616,1.8]
  #labels_comp = ['comp_bottom_10','comp_middle_80','comp_top_10']
  #hclv['Comp_Binned'] = pd.cut(hclv['Compa Ratio'], bins=bins_comp, labels=labels_comp, include_lowest=True).astype("object")

  ### DISTANCE
  hclv['Office_Commute_Distance'].quantile([0.25,0.50,0.75])
  bins_dist = [0,6.67,13.82,25.32,11905]
  labels_dist = ['dist_Q1','dist_Q2','dist_Q3','dist_Q4']
  hclv['Dist_Binned'] = pd.cut(hclv['Office_Commute_Distance'], bins=bins_dist, labels=labels_dist, include_lowest=True).astype("object")
  
  return hclv
  
#--------------------------------------------------------------------------------------------------------

def append_historical_info(hclv,snap_date):
  """This function will append all the historical info for column entered in the function."""
  # Read hc file and drop IDs
  import os
  os.chdir("/home/cdsw/Data")
  
  hc = pd.read_csv("merged_headcount_WHR_V4.csv")
  
  # only keep the records before the snap date
  hc = hc[pd.to_datetime(hc['Report Date']) < snap_date]
  
  hc = hc.drop_duplicates()
  with open('drop_id_list.pkl','rb') as f:
    drop_ID=pickle.load(f)
  hc = hc[~hc.Employee_ID.isin(drop_ID)]
  
   # Clean up titles
  titles=['Analyst','Assoc. Analyst','Manager','Assoc. Manager','Sr. Manager',
           'Group Lead','Director','Assoc. Director','Sr. Director','Coordinator',
           'VP','Assoc. VP','Sr. VP','Global Non Banded','Others','Global - Hourly Others']
  hc['Title_Grouped']=''
  for i in titles:
    hc['Title_Grouped'][hc.Title.str.contains(i)]=i
    
  # Loop across columns
  L1 = ['Level','Function','Job_Band','Division','Title_Grouped']
  for col in L1:
    hclv[col + '_Hist'] = hclv['Employee_ID'].map(hc.groupby('Employee_ID').apply(lambda x:','.join(x[col])))
  
  L2 = ['NewBusinessUnit','BUCluster']
  for col in L2:
    hclv[col + '_Hist'] = hclv['Employee_ID'].map(hc.groupby('Employee_ID').apply(lambda x:','.join(x[col].astype(str))))
  
  # Gender and Ethnicity are coded in a similar way. Will replace Gender column to be G1, G2, G3
  val_to_replace = {'Group 1':'G1','Group 2':'G2','Group 3':'G3','Others':'Other'}
  hclv['Gender']=hclv['Gender'].map(val_to_replace)
  
  return hclv

#------------------------------------------------------------------------------------------------------------

def bag_of_words(hclv):
  '''This function selects the columns to be used for content-based recommendation model, and creates a dataframe with employee
  and bag-of-words describing each employee from these columns'''

  col_to_use_for_cb=['Employee_ID','Work Address - Country','Work_City_Grouped','Marital_Status','Gender','Highest Degree',
                  'Work Address - State/Province','Pay Rate Type',
                  'Region_x','Age_Bucket','Ethnicity','Level_Hist','Function_Hist','Job_Band_Hist',
                   'Division_Hist','Title_Grouped_Hist','YOS_Binned',
                  'Dist_Binned','Term_Reason_Final','BUCluster_Hist','NewBusinessUnit_Hist']
  
  hclv=hclv[col_to_use_for_cb]
  hclv=hclv[hclv.Term_Reason_Final != 'Involuntary']
  
  # Strip all white spaces
  hclv = hclv.apply(lambda x: x.str.replace(" ","") if x.dtype == 'object' else x)

  # Create bag-of-words
  hclv["Highest Degree"]=hclv["Highest Degree"].str.replace('\d+','').str.replace('.','')
  hclv['combined']=hclv[['Work Address - Country','Work_City_Grouped','Marital_Status','Gender','Highest Degree',
                  'Work Address - State/Province','Pay Rate Type',
                  'Region_x','Age_Bucket','Ethnicity','Level_Hist','Function_Hist','Job_Band_Hist',
                   'Division_Hist','Title_Grouped_Hist','YOS_Binned',
                  'Dist_Binned','BUCluster_Hist','NewBusinessUnit_Hist']].apply(lambda row: ' '.join(row.values.astype(str)),axis=1)
  

  return hclv

#--------------------------------------------------------------------------------------------------------------
  
# Final CB dataframe

def data_processing(snap_date = '2019-12-01'):
  
  from hclv import create_hclv
  
  hclv = create_hclv(snap_date)

  hclv = create_grouped_columns(hclv)
  
  hclv = drop_columns(hclv)

  hclv = missing_values(hclv)
  
  hclv = numerical_to_bins(hclv)
  
  hclv = append_historical_info(hclv,snap_date)
  
  original_hclv = hclv.copy()
  
  hclv = bag_of_words(hclv)
  
  return hclv, original_hclv


if __name__ == '__main__':
  main()
