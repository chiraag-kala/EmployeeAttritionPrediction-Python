import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import re
import os

# --- Cleaning Functions

def create_grouped_columns(hclv):
  """This function creates custom groupings for any high-dimensional columns."""
  
  #'Work Address - City' --> Work_City_Grouped 
  cities_to_group = list(hclv['Work Address - City'].value_counts()[hclv['Work Address - City'].value_counts()<=10].index)
  hclv['Work_City_Grouped'] = hclv['Work Address - City']
  hclv['Work_City_Grouped'][hclv['Work_City_Grouped'].isin(cities_to_group)] = 'Other'
  hclv['Work_City_Grouped'].value_counts()
  
  return hclv


def drop_columns(hclv):
  """This function drops columns that won't be used for the ML Model."""
  
  all_columns = list(hclv.columns)
  
  columns_to_remove = [
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
                       'MBO_17_Combined',
                       'MBO_18_Combined',
                       '9-Box 17',
                       '9-Box 18',
                       'Most_Recent_9_Box',
                       'TGroup_Rep_diff',
                       'Min_Diff',
                       'Term_Group']

  high_dim_cols = ['Work Address - City',
                   'Primary Work Address',
                   'Location_x']

  leaver_cols = ['Country',
                 'Business_Title',
                 'Employee Type',
                 'Job Category',
                 'Global Comp Ratio',
                 'Term Reason',
                 'Length of Service',
                 'Termination Month',
                 'Termination Year']

  historical_append_cols = ['Level_x',
                            'Function_x',
                            'NewBusinessUnit_x',
                            'Division_x',
                            'Job_Band_x',
                            'Title',
                            'Title_Grouped']

  final_columns = set(all_columns) - set(high_dim_cols) - set(columns_to_remove) - set(leaver_cols) - set(historical_append_cols)
  
  return hclv[final_columns]
 
  
def append_historical_info(col_name,hclv):
  """This function will append all the historical info for column entered in the function."""
  """Level, Function, Job_Band, Title_Grouped, Division"""
  import os
  os.chdir("/home/cdsw/Data")
  
  hc = pd.read_csv("merged_headcount_WHR_V4.csv")
  hc = hc.drop_duplicates()
  with open('drop_id_list.pkl','rb') as f:
    drop_ID=pickle.load(f)
  hc = hc[~hc.Employee_ID.isin(drop_ID)]
  columns_to_drop = ['Staffing Status','Position Worker Type','Active Status','CF - Wrkr Non Exempt','Job Category']
  hc=hc.drop(columns_to_drop, axis=1)
  hc['Report Date']=pd.to_datetime(hc['Report Date'])
  
  titles=['Analyst','Assoc. Analyst','Manager','Assoc. Manager','Sr. Manager',
           'Group Lead','Director','Assoc. Director','Sr. Director','Coordinator',
           'VP','Assoc. VP','Sr. VP','Global Non Banded','Others','Global - Hourly Others']
  hc['Title_Grouped']=''
  for i in titles:
    hc['Title_Grouped'][hc.Title.str.contains(i)]=i
  
  hc_2 = pd.merge(hc,hclv[['Employee_ID','Report Date']],how='left',left_on = "Employee_ID", right_on= "Employee_ID")
  hc_2 = hc_2[hc_2['Report Date_x'] <= hc_2['Report Date_y']]
 
  cols_to_join = hc_2.groupby(['Employee_ID',col_name])[col_name].nunique().unstack().fillna(0).reset_index()
  hclv = pd.merge(hclv,cols_to_join,how='inner',on='Employee_ID')
  
  return hclv


def missing_values(hclv):
  """This function imputes any missing values."""
  #BUCluster
  hclv['BUCluster_x'].fillna(hclv['BUCluster_x'].mode()[0],inplace=True)
  #hclv['NewBusinessUnit_x'].fillna(hclv['NewBusinessUnit_x'].mode()[0],inplace=True)
  
  #Demographic Info
  hclv['Gender'].fillna(hclv.Gender.mode(),inplace=True)
  hclv['Highest Degree'] = hclv.groupby('BUCluster_x')['Highest Degree'].transform(lambda x: x.fillna(x.mode()[0]))
  hclv['Ethnicity'].fillna( hclv['Ethnicity'].mode()[0],inplace=True)
  hclv['Work_City_Grouped'].fillna("Other",inplace=True)
  hclv['Marital_Status'].fillna(hclv['Marital_Status'].mode()[0],inplace=True)
  
  #Office_Commute_Distance
  hclv['Office_Commute_Distance'] = hclv.groupby('BUCluster_x')['Office_Commute_Distance'].transform(lambda x: x.fillna(x.median()))
  #This one person has his own BUCluster, and has NaN value
  hclv['Office_Commute_Distance'].fillna(hclv['Office_Commute_Distance'].median(),inplace=True)
  
  #MBO Score
  hclv['Most_Recent_MBO'] = hclv.groupby('BUCluster_x')['Most_Recent_MBO'].transform(lambda x: x.fillna(x.median()))
  #There are 16 people whose whole BUCluster misses MBO Score
  hclv['Most_Recent_MBO'].fillna(hclv['Most_Recent_MBO'].median(),inplace=True)
  
  #Manager_Employee_Count
  hclv['Manager_Employee_Count'].fillna(0,inplace=True)
  
  #Merit
  #merit_cols = list(hclv.columns[hclv.columns.str.contains('merit')])
  
  #for col in merit_cols:
    #hclv[col].fillna(0,inplace=True)
  
  #Creating an Interaction Variable
  #hclv['MBO_X_CompaRatio'] = hclv['Most_Recent_MBO']/100.00 * hclv['Compa Ratio']
  
  return hclv
  
def encode_and_clean(hclv):
  """This function takes remaining cleaning steps and then dummy encodes all variables. """
  #Drop Involuntary + Term_Reason_Final
  
  #hclv = hclv[hclv['Term_Reason_Final'] != 'Involuntary']
  hclv = hclv.drop(['Term_Reason_Final'],axis=1)
  
  emp_IDs = hclv['Employee_ID']
  
  #Drop Employee_ID
  hclv = hclv.drop(['Employee_ID'],axis=1)
  hclv = hclv.drop(['Report Date'],axis=1)
  
  hclv = pd.get_dummies(hclv)
  
  print('Shape Before Emp_IDs:',hclv.shape)

  #Join Emp ID Back 
  hclv = pd.merge(hclv,emp_IDs,left_index=True,right_index=True)
  
  print('Shape After Emp_IDs:',hclv.shape)
  
  #Remove <
  hclv = hclv.rename(columns={'Age_Bucket_<25':"Age_Bucket_Under_25"})
  
  return hclv 


# --- Final ML DF

def main():
    
  from hclv import create_hclv_tb
  
  hclv_tb = create_hclv_tb()
  
  print(hclv_tb.shape)

  hclv_tb = create_grouped_columns(hclv_tb)
  
  print(hclv_tb.shape)
  
  hclv_tb = drop_columns(hclv_tb)
  
  print(hclv_tb.shape)
  
  hist_columns = ['Level','Function','Division','Job_Band','Title_Grouped','NewBusinessUnit']
  for col in hist_columns:
    hclv_tb = append_historical_info(col,hclv_tb)
  
  print(hclv_tb.shape)

  hclv_tb = missing_values(hclv_tb)
  
  print(hclv_tb.shape)
  
  hclv_tb = encode_and_clean(hclv_tb)
  
  print('FINAL SHAPE:',hclv_tb.shape)
  
  return hclv_tb

if __name__ == '__main__':
  main()


  
