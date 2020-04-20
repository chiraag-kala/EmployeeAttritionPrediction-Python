"""
HCLV - Creates HCLV with a snap_date argument. 
"""

def create_hclv(snap_date = '2019-12-01'):
  
  """ This function will return the dataframe with a unique record for each White Collar, Full-Time
  employee and include the most recent information for each employeee. 
  It will also include MBO Scores and OPR from 2017 and 2018.

  It will include several new columns 
  - Title Grouped
  - Counts of Level, Job Bands, etc.
  - MBO Self + Team Combined for 2017 and 2018 
  """
  
  import os
  os.chdir("/home/cdsw/Data")

  import pandas as pd
  import numpy as np
  import pickle
  import sys
  import re
  
  #Loading Combined HC File and Dropping Duplicates/Other Uninformative Columns
  hc = pd.read_csv("merged_headcount_WHR_V4.csv")
                
  # only keep the records before the snap date
  hc = hc[pd.to_datetime(hc['Report Date']) <= snap_date]
                
  hc = hc.drop_duplicates()
  
  # Drop IDs that have unexplainable duplicates in each month
  record_count = hc.groupby(['Employee_ID','Report Date'])['Active Status'].count().reset_index()
  IDs_to_drop = record_count[record_count['Active Status'] > 1]["Employee_ID"].unique()
  hc = hc[~hc.Employee_ID.isin(IDs_to_drop)]
  
  # Drop columns of no use
  columns_to_drop = ['Staffing Status','Active Status','CF - Wrkr Non Exempt','Job Category']
  hc=hc.drop(columns_to_drop, axis=1)
  
  #Grouping Title
  titles=['Analyst','Assoc. Analyst','Manager','Assoc. Manager','Sr. Manager',
           'Group Lead','Director','Assoc. Director','Sr. Director','Coordinator',
           'VP','Assoc. VP','Sr. VP','Global Non Banded','Others','Global - Hourly Others']
  hc['Title_Grouped']=''
  for i in titles:
    hc['Title_Grouped'][hc.Title.str.contains(i)]=i
  
  
  #Converting Columns and Creating New Columns 
  hc['Report Date']=pd.to_datetime(hc['Report Date'])
  hc['Unique_Level_Changes'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Level'].nunique())
  hc['Position_Count'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Title'].nunique())
  hc['Function_Count'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Function'].nunique())
  hc['Division_Count'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Division'].nunique())
  hc['BU_Count'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['NewBusinessUnit'].nunique())
  hc['Band_Count'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Job_Band'].nunique())
  hc['Manager Switches'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Manager Employee_ID'].nunique())
  
  #Upward/Downward Counts
  level_hist = hc['Employee_ID'].map(hc.groupby('Employee_ID').apply(lambda x:','.join(x['Level'].str.replace('L',''))))
  #Upward Level Counts
  hc['Downward_Level_Counts'] = level_hist.apply(lambda x: np.count_nonzero(np.diff(np.fromstring(x,sep=','))>0))
  #Downward Level Counts
  hc['Upward_Level_Counts'] = level_hist.apply(lambda x: np.count_nonzero(np.diff(np.fromstring(x,sep=','))<0))

  #Keeping Mosty Recent Record Based on Report Date
  hc = hc.sort_values(by = ['Report Date'])
  hc.drop_duplicates(subset = ['Employee_ID'], keep = 'last', inplace = True)
  
  #Importing Leavers Report
  lv = pd.read_csv("leavers_WHR_V4.csv")
  lv['Term_Reason_Final'] = lv['Term Reason'].apply(lambda x: re.findall('> (.*?) >',str(x)))
  lv['Term_Reason_Final'] = lv['Term_Reason_Final'].apply(lambda x: "".join(x))
  
  #Joining Leavers Report and Creating Churn Column
  hclv = hc.merge(lv, left_on = "Employee_ID", right_on = "Employee_ID", how = "left")
  hclv = hclv.drop(hclv.columns[hclv.columns.str.contains("_y")],axis=1)
  hclv['Churn'] = hclv['Termination Date'].isnull().apply(lambda x: 0 if x==True else 1)

  #Dropping instances of ['Retirement','Severance Reasons','Death in Service','Did Not Start'] 
  hclv = hclv[~hclv['Term_Reason_Final'].isin(['Retirement','Severance Reasons','Death in Service','Did Not Start'])]
  hclv.Term_Reason_Final = hclv.Term_Reason_Final.fillna('Active')
  hclv.Term_Reason_Final.value_counts()
  
  #Dropping IDs with incomplete records.
  
  #Case 1. If a person is still active, but his/her latest report date is not up-to-date, this ID
  #is considered to be imcomplete.  
  
  active = hclv[hclv.Churn == 0][['Employee_ID','Report Date']].reset_index(drop = True)
  active_to_drop = active[active['Report Date'] < active['Report Date'].max()]["Employee_ID"].unique()
  hclv = hclv[~hclv.Employee_ID.isin(active_to_drop)]
  
  #Case 2. If a person has left in a certain month, but his/her last report is 2 month or more before/after
  #the actual termination date, this ID is considered to be incomplete or incorrect
  
  churned = hclv[hclv.Churn == 1][['Employee_ID','Report Date','Termination Date']].reset_index(drop = True)
  churned['Termination Date']=pd.to_datetime(churned['Termination Date'])
  churned["Days_difference"] = (churned['Termination Date'] - churned['Report Date']).dt.days
  incorrect_IDs = churned[churned.Days_difference < 0]['Employee_ID'].unique()
  incomplete_IDs = churned[churned.Days_difference > 62]['Employee_ID'].unique()
  hclv = hclv[~hclv.Employee_ID.isin(incorrect_IDs)]
  hclv = hclv[~hclv.Employee_ID.isin(incomplete_IDs)]
  
  #Contagious Manager
  #Count Number of Employees Under Manager
  Employee_Manager_Count = hclv.groupby('Manager Employee_ID').count()[['Employee_ID']].reset_index()
  Employee_Manager_Count.rename(columns={'Employee_ID':"Manager_Employee_Count"},inplace=True)
  hclv = hclv.merge(Employee_Manager_Count, how='left', on='Manager Employee_ID')

  #Joining MBO
  MBO_2017=pd.read_excel("MBO_Scores.xlsx",sheet_name='2017')
  MBO_2018=pd.read_excel("MBO_Scores.xlsx",sheet_name='2018')
  Team_17=MBO_2017[MBO_2017.Type=="Team"].drop("Type",axis=1)
  Team_17.rename(columns={"SCORE":"MBO_Team_17"},inplace=True)
  Self_17=MBO_2017[MBO_2017.Type=="Self"].drop("Type",axis=1)
  Self_17.rename(columns={"SCORE":"MBO_Self_17"},inplace=True)
  Team_18=MBO_2018[MBO_2018.Type=="Team"].drop("Type",axis=1)
  Team_18.rename(columns={"SCORE":"MBO_Team_18"},inplace=True)
  Self_18=MBO_2018[MBO_2018.Type=="Self"].drop("Type",axis=1)
  Self_18.rename(columns={"SCORE":"MBO_Self_18"},inplace=True)
  hclv = pd.merge(hclv, Team_17, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)
  hclv = pd.merge(hclv, Self_17, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)
  hclv = pd.merge(hclv, Team_18, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)
  hclv = pd.merge(hclv, Self_18, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)

  hclv['MBO_17_Combined'] = hclv.MBO_Self_17.fillna(hclv.MBO_Team_17)
  hclv['MBO_18_Combined'] = hclv.MBO_Self_18.fillna(hclv.MBO_Team_18)
  hclv['Most_Recent_MBO'] = np.where(hclv['Termination Year'] == 2017,hclv.MBO_17_Combined,hclv.MBO_18_Combined)
  hclv.Most_Recent_MBO = pd.to_numeric(hclv.Most_Recent_MBO)
  
  #Joining OPR
  OPR = pd.read_excel("OPR Data.xlsx")
  OPR = OPR.drop_duplicates()
  OPR_17 = OPR[OPR.Year == 2017]
  OPR_18 = OPR[OPR.Year == 2018]
  OPR_17 = OPR_17.drop(OPR_17[OPR_17['9-Box Score']=="Newcomer"].index)
  OPR_17 = OPR_17.drop("Year",axis=1)
  OPR_17.rename(columns={"9-Box Score":"9-Box 17"},inplace=True)
  OPR_18 = OPR_18.drop(OPR_18[OPR_18['9-Box Score']=="Newcomer"].index)
  drop_ID = ('e970dacf','590910e8','b5ac5083','1a8846ee')
  OPR_18 = OPR_18[~OPR_18.Employee_ID.isin(drop_ID)]
  OPR_18=OPR_18.dropna()
  OPR_18 = OPR_18.drop("Year",axis=1)
  OPR_18.rename(columns={"9-Box Score":"9-Box 18"},inplace=True)
  OPR_HR = pd.read_excel("OPR_additional.xlsx",sheet_name='HR')
  OPR_LOGI = pd.read_excel("OPR_additional.xlsx",sheet_name='Logistics')
  OPR_MFG = pd.read_excel("OPR_additional.xlsx",sheet_name='Mfg')
  OPR_PROC = pd.read_excel("OPR_additional.xlsx",sheet_name='Proc')
  OPR_HR=OPR_HR.dropna()
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2017']=="New comer"].index)
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2018']=="New comer"].index)
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2017']=='0'].index)
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2018']=='0'].index)
  OPR_HR_17=OPR_HR[['Emp ID','2017']]
  OPR_HR_17.rename(columns={"2017":"9-Box 17","Emp ID":"Employee_ID"},inplace=True)
  OPR_HR_18=OPR_HR[['Emp ID','2018']]
  OPR_HR_18.rename(columns={"2018":"9-Box 18","Emp ID":"Employee_ID"},inplace=True)
  OPR_LOGI=OPR_LOGI.dropna()
  OPR_LOGI_17=OPR_LOGI[['EMP ID','9 Box - 2017']]
  OPR_LOGI_17=OPR_LOGI_17.drop(OPR_LOGI_17[OPR_LOGI_17['9 Box - 2017']=="1B/2B"].index)
  OPR_LOGI_17.rename(columns={"9 Box - 2017":"9-Box 17","EMP ID":"Employee_ID"},inplace=True)
  OPR_LOGI_18=OPR_LOGI[['EMP ID','9 Box - 2018']]
  OPR_LOGI_18.rename(columns={"9 Box - 2018":"9-Box 18","EMP ID":"Employee_ID"},inplace=True)
  OPR_MFG=OPR_MFG.dropna()
  row_drop = ("New to role/org (<3 months)","No rated")
  OPR_MFG=OPR_MFG[~OPR_MFG['2017'].isin(row_drop)]
  OPR_MFG=OPR_MFG[~OPR_MFG['2018'].isin(row_drop)]
  OPR_MFG["2017"]=OPR_MFG["2017"].str.replace("2a","2A")
  OPR_MFG["2017"]=OPR_MFG["2017"].str.replace("3a","3A")
  OPR_MFG_17=OPR_MFG[['Emp Id','2017']]
  OPR_MFG_17.rename(columns={"2017":"9-Box 17","Emp Id":"Employee_ID"},inplace=True)
  OPR_MFG_18=OPR_MFG[['Emp Id','2018']]
  OPR_MFG_18.rename(columns={"2018":"9-Box 18","Emp Id":"Employee_ID"},inplace=True)
  OPR_PROC=OPR_PROC.dropna()
  OPR_PROC_17=OPR_PROC[['Emp ID','9 Box - 2017']]
  OPR_PROC_17.rename(columns={"9 Box - 2017":"9-Box 17","Emp ID":"Employee_ID"},inplace=True)
  OPR_PROC_18=OPR_PROC[['Emp ID','9 Box - 2018 (June)']]
  OPR_PROC_18.rename(columns={"9 Box - 2018 (June)":"9-Box 18","Emp ID":"Employee_ID"},inplace=True)
  OPR_17_all=OPR_17.append([OPR_HR_17,OPR_LOGI_17,OPR_MFG_17,OPR_PROC_17]).drop_duplicates()
  OPR_18_all=OPR_18.append([OPR_HR_18,OPR_LOGI_18,OPR_MFG_18,OPR_PROC_18]).drop_duplicates()
  hclv=pd.merge(hclv, OPR_17_all, left_on="Employee_ID",right_on="Employee_ID",how="left")
  hclv=pd.merge(hclv, OPR_18_all, left_on="Employee_ID",right_on="Employee_ID",how="left")

  hclv['Most_Recent_9_Box'] = np.where(hclv['Termination Year'] == 2017,hclv['9-Box 17'],hclv['9-Box 18'])
  
  #Adding Commute Distance
  ## Read in the 4 zipcodes dataframes from the pickle files
  with open('act_zip_us.pickle','rb') as f1:
    pk_act_zip_us=pickle.load(f1)
  with open('act_zip_ca.pickle','rb') as f2:
    pk_act_zip_ca=pickle.load(f2)
  with open('term_zip_us.pickle','rb') as f3:
    pk_term_zip_us=pickle.load(f3)
  with open('term_zip_ca.pickle','rb') as f:
    pk_term_zip_ca=pickle.load(f)
  ## Merge all 4 Df into 1
  df_zip = pk_act_zip_us.append(pk_act_zip_ca).append(pk_term_zip_us).append(pk_term_zip_ca)
  df_zip = df_zip.drop_duplicates()
  df_zip = df_zip[['Employee_ID','distance']].reset_index().drop("index",axis=1)
  df_zip.rename(columns={"distance":"Office_Commute_Distance"},inplace=True)
  hclv=pd.merge(hclv,df_zip,how='left',on='Employee_ID')
  hclv.Office_Commute_Distance = pd.to_numeric(hclv.Office_Commute_Distance)
  
  #Dropping Duplicates created by OPR + Distance Join
  hclv = hclv.sort_values('9-Box 18').drop_duplicates(subset=['Employee_ID'],keep='last')
 
  os.chdir("/home/cdsw")
  
  from merit import create_merit
  merit = create_merit(snap_date)
  hclv = pd.merge(hclv, merit,how='left',left_on = "Employee_ID", right_on= "merit_Employee_ID")

  hclv = hclv.drop(['merit_Employee_ID', 'merit_Effective_Date'], axis=1)
  
  
  #- ECONOMIC INDICATORS 
  hclv["Adjusted Effective Date"] = pd.to_datetime(hclv['Effective Date for Current Position_x'])
  hclv["Adjusted Effective Date"] = np.where(hclv["Adjusted Effective Date"] < "2016-01-01","2016-01-01",hclv['Effective Date for Current Position_x'])

  os.chdir("/home/cdsw")
  from economic_indicators import Create_Macro
  Full_Macro = Create_Macro(0)

  #Make Effective Month and Year Columns
  hclv['Effective Date'] = pd.to_datetime(hclv['Adjusted Effective Date'])
  hclv['Effective Month'] = hclv['Effective Date'].dt.month
  hclv['Effective Year'] = hclv['Effective Date'].dt.year

  #Merge with Macro for Effective
  hclv = hclv.merge(Full_Macro, how='left', left_on=['Effective Month', 'Effective Year'], right_on=['Month','Year'])
  hclv.rename(columns={'CPI_Urban_CSL_Shifted':'CPI_Urban_CSL_Effective', 'Unemployment_Rate_Shifted':'Unemployment_Rate_Effective', 'PPI_Finished_Goods_Shifted':'PPI_Finished_Goods_Effective', 'CCI_Shifted':'CCI_Effective',
                        'Average_Hourly_Earnings_Shifted':'Average_Hourly_Earnings_Effective', 'Real_GDP_Quarterly_Shifted':'Real_GDP_Quarterly_Effective'},inplace=True)

  #Remove Time Based Columns
  hclv = hclv.drop(['Effective Month','Effective Year'], axis = 1)

  #Drop Additional Columns
  hclv = hclv.drop(['Real_GDP_Quarterly_Effective', "Adjusted Effective Date",'Effective Date','Average_Hourly_Earnings_Effective','CPI_Urban_CSL_Effective',
                    'PPI_Finished_Goods_Effective','Unemployment_Rate_Effective'], axis = 1)
  
  
  
  # Return completed hclv Dataframe
  return hclv



"""
HCLV - Time-Based Function

This function creates HCLV with the 6 Checkpoints Used in the ML Model.
"""

def create_hclv_tb():
  
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pickle
  import numpy as np
  import re
  import os
  
  #Importing HC
  os.chdir("/home/cdsw/Data")
  hc = pd.read_csv("merged_headcount_WHR_V4.csv")
  hc = hc.drop_duplicates()
 
  # Drop IDs that have unexplainable duplicates in each month
  record_count = hc.groupby(['Employee_ID','Report Date'])['Active Status'].count().reset_index()
  IDs_to_drop = record_count[record_count['Active Status'] > 1]["Employee_ID"].unique()
  hc = hc[~hc.Employee_ID.isin(IDs_to_drop)]

  columns_to_drop = ['Staffing Status','Active Status','CF - Wrkr Non Exempt','Job Category']
  hc=hc.drop(columns_to_drop, axis=1)
  hc['Report Date']=pd.to_datetime(hc['Report Date'])
  #hc['Most_Recent_Report_Date'] = hc['Employee_ID'].map(hc.groupby(['Employee_ID'])['Report Date'].max())
  
  
  #Importing Leavers Report
  lv = pd.read_csv("leavers_WHR_V4.csv")
  lv['Term_Reason_Final'] = lv['Term Reason'].apply(lambda x: re.findall('> (.*?) >',str(x)))
  lv['Term_Reason_Final'] = lv['Term_Reason_Final'].apply(lambda x: "".join(x))

  #Joining Leavers Report and Creating Churn Column
  hclv = hc.merge(lv, left_on = "Employee_ID", right_on = "Employee_ID", how = "left")
  hclv = hclv.drop(hclv.columns[hclv.columns.str.contains("_y")],axis=1)
  hclv['Churn'] = hclv['Termination Date'].isnull().apply(lambda x: 0 if x==True else 1)
  hclv['Termination Date']=pd.to_datetime(hclv['Termination Date'])
  
  #Dropping instances of ['Retirement','Severance Reasons','Death in Service','Did Not Start'] 
  hclv = hclv[~hclv['Term_Reason_Final'].isin(['Retirement','Severance Reasons','Death in Service','Did Not Start'])]
  hclv.Term_Reason_Final = hclv.Term_Reason_Final.fillna('Active')
  hclv.Term_Reason_Final.value_counts()
  
  #Removing Involuntary Churn
  hclv = hclv[hclv['Term_Reason_Final'] != 'Involuntary']

  #Creating Term Groups
  hclv['Term_Group'] = '2019-10-01'
  hclv.loc[((hclv['Termination Date'] >= '2017-01-01') & (hclv['Termination Date'] <= '2017-06-30')),'Term_Group'] = '2017-01-01'
  hclv.loc[((hclv['Termination Date'] > '2017-06-30') & (hclv['Termination Date'] <= '2017-12-31')),'Term_Group'] = '2017-07-01'
  hclv.loc[((hclv['Termination Date'] > '2017-12-31') & (hclv['Termination Date'] <= '2018-06-30')),'Term_Group'] = '2018-01-01'
  hclv.loc[((hclv['Termination Date'] > '2018-06-30') & (hclv['Termination Date'] <= '2018-12-31')),'Term_Group'] = '2018-07-01'
  hclv.loc[((hclv['Termination Date'] > '2018-12-31') & (hclv['Termination Date'] <= '2019-06-30')),'Term_Group'] = '2019-01-01'
  hclv.loc[(hclv['Termination Date'] > '2019-06-30'),'Term_Group'] = '2019-07-01'
  hclv['Term_Group']=pd.to_datetime(hclv['Term_Group'])

  hclv['TGroup_Rep_diff'] = abs(hclv['Term_Group'] - hclv['Report Date'])
  hclv['Min_Diff'] = hclv['Employee_ID'].map(hclv.groupby('Employee_ID')['TGroup_Rep_diff'].min())

  hclv = hclv[hclv['Min_Diff'] == hclv['TGroup_Rep_diff']]
  hclv = hclv[hclv['TGroup_Rep_diff'] <= '31 days']

  #Creating Grouped Columns as of Report Date
  hc_2 = pd.merge(hc,hclv[['Employee_ID','Report Date']],how='left',left_on = "Employee_ID", right_on= "Employee_ID")
  hc_2 = hc_2[hc_2['Report Date_x'] <= hc_2['Report Date_y']]

  hc_2['Unique_Level_Changes'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['Level'].nunique())
  hc_2['Position_Count'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['Title'].nunique())
  hc_2['Function_Count'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['Function'].nunique())
  hc_2['Division_Count'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['Division'].nunique())
  hc_2['BU_Count'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['NewBusinessUnit'].nunique())
  hc_2['Band_Count'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['Job_Band'].nunique())
  hc_2['Manager Switches'] = hc_2['Employee_ID'].map(hc_2.groupby(['Employee_ID'])['Manager Employee_ID'].nunique())
  
  #Upward/Downward Counts
  hc_2['Level_Append'] = hc_2['Employee_ID'].map(hc_2.groupby('Employee_ID').apply(lambda x:','.join(x['Level'].str.replace('L',''))))
  #Upward Level Counts
  hc_2['Downward_Level_Counts'] = hc_2['Level_Append'].apply(lambda x: np.count_nonzero(np.diff(np.fromstring(x,sep=','))>0))
  #Downward Level Counts
  hc_2['Upward_Level_Counts'] = hc_2['Level_Append'].apply(lambda x: np.count_nonzero(np.diff(np.fromstring(x,sep=','))<0))

  
  hc_2 = hc_2.sort_values(by = ['Report Date_x'])
  hc_2.drop_duplicates(subset = ['Employee_ID'], keep = 'last', inplace = True)

  hc_counts = hc_2[['Employee_ID','Unique_Level_Changes','Downward_Level_Counts','Upward_Level_Counts','Position_Count', 'Function_Count', 'Division_Count', 'BU_Count',
                    'Band_Count', 'Manager Switches']]

  hclv = pd.merge(hclv,hc_counts,how='left',left_on = "Employee_ID", right_on= "Employee_ID")

  
  #Contagious Manager
  #Count Number of Employees Under Manager
  Employee_Manager_Count = hclv.groupby('Manager Employee_ID').count()[['Employee_ID']].reset_index()
  Employee_Manager_Count.rename(columns={'Employee_ID':"Manager_Employee_Count"},inplace=True)
  hclv = hclv.merge(Employee_Manager_Count, how='left', on='Manager Employee_ID')
  
  #Joining MBO
  MBO_2017=pd.read_excel("MBO_Scores.xlsx",sheet_name='2017')
  MBO_2018=pd.read_excel("MBO_Scores.xlsx",sheet_name='2018')
  Team_17=MBO_2017[MBO_2017.Type=="Team"].drop("Type",axis=1)
  Team_17.rename(columns={"SCORE":"MBO_Team_17"},inplace=True)
  Self_17=MBO_2017[MBO_2017.Type=="Self"].drop("Type",axis=1)
  Self_17.rename(columns={"SCORE":"MBO_Self_17"},inplace=True)
  Team_18=MBO_2018[MBO_2018.Type=="Team"].drop("Type",axis=1)
  Team_18.rename(columns={"SCORE":"MBO_Team_18"},inplace=True)
  Self_18=MBO_2018[MBO_2018.Type=="Self"].drop("Type",axis=1)
  Self_18.rename(columns={"SCORE":"MBO_Self_18"},inplace=True)
  hclv = pd.merge(hclv, Team_17, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)
  hclv = pd.merge(hclv, Self_17, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)
  hclv = pd.merge(hclv, Team_18, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)
  hclv = pd.merge(hclv, Self_18, left_on="Employee_ID",right_on="EMPLOYEE ID",how="left").drop("EMPLOYEE ID",axis=1)

  hclv['MBO_17_Combined'] = hclv.MBO_Self_17.fillna(hclv.MBO_Team_17)
  hclv['MBO_18_Combined'] = hclv.MBO_Self_18.fillna(hclv.MBO_Team_18)
  hclv['Most_Recent_MBO'] = np.where(hclv['Termination Year'] == 2017,hclv.MBO_17_Combined,hclv.MBO_18_Combined)
  hclv.Most_Recent_MBO = pd.to_numeric(hclv.Most_Recent_MBO)
  
  #Joining OPR
  OPR = pd.read_excel("OPR Data.xlsx")
  OPR = OPR.drop_duplicates()
  OPR_17 = OPR[OPR.Year == 2017]
  OPR_18 = OPR[OPR.Year == 2018]
  OPR_17 = OPR_17.drop(OPR_17[OPR_17['9-Box Score']=="Newcomer"].index)
  OPR_17 = OPR_17.drop("Year",axis=1)
  OPR_17.rename(columns={"9-Box Score":"9-Box 17"},inplace=True)
  OPR_18 = OPR_18.drop(OPR_18[OPR_18['9-Box Score']=="Newcomer"].index)
  drop_ID = ('e970dacf','590910e8','b5ac5083','1a8846ee')
  OPR_18 = OPR_18[~OPR_18.Employee_ID.isin(drop_ID)]
  OPR_18=OPR_18.dropna()
  OPR_18 = OPR_18.drop("Year",axis=1)
  OPR_18.rename(columns={"9-Box Score":"9-Box 18"},inplace=True)
  OPR_HR = pd.read_excel("OPR_additional.xlsx",sheet_name='HR')
  OPR_LOGI = pd.read_excel("OPR_additional.xlsx",sheet_name='Logistics')
  OPR_MFG = pd.read_excel("OPR_additional.xlsx",sheet_name='Mfg')
  OPR_PROC = pd.read_excel("OPR_additional.xlsx",sheet_name='Proc')
  OPR_HR=OPR_HR.dropna()
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2017']=="New comer"].index)
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2018']=="New comer"].index)
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2017']=='0'].index)
  OPR_HR=OPR_HR.drop(OPR_HR[OPR_HR['2018']=='0'].index)
  OPR_HR_17=OPR_HR[['Emp ID','2017']]
  OPR_HR_17.rename(columns={"2017":"9-Box 17","Emp ID":"Employee_ID"},inplace=True)
  OPR_HR_18=OPR_HR[['Emp ID','2018']]
  OPR_HR_18.rename(columns={"2018":"9-Box 18","Emp ID":"Employee_ID"},inplace=True)
  OPR_LOGI=OPR_LOGI.dropna()
  OPR_LOGI_17=OPR_LOGI[['EMP ID','9 Box - 2017']]
  OPR_LOGI_17=OPR_LOGI_17.drop(OPR_LOGI_17[OPR_LOGI_17['9 Box - 2017']=="1B/2B"].index)
  OPR_LOGI_17.rename(columns={"9 Box - 2017":"9-Box 17","EMP ID":"Employee_ID"},inplace=True)
  OPR_LOGI_18=OPR_LOGI[['EMP ID','9 Box - 2018']]
  OPR_LOGI_18.rename(columns={"9 Box - 2018":"9-Box 18","EMP ID":"Employee_ID"},inplace=True)
  OPR_MFG=OPR_MFG.dropna()
  row_drop = ("New to role/org (<3 months)","No rated")
  OPR_MFG=OPR_MFG[~OPR_MFG['2017'].isin(row_drop)]
  OPR_MFG=OPR_MFG[~OPR_MFG['2018'].isin(row_drop)]
  OPR_MFG["2017"]=OPR_MFG["2017"].str.replace("2a","2A")
  OPR_MFG["2017"]=OPR_MFG["2017"].str.replace("3a","3A")
  OPR_MFG_17=OPR_MFG[['Emp Id','2017']]
  OPR_MFG_17.rename(columns={"2017":"9-Box 17","Emp Id":"Employee_ID"},inplace=True)
  OPR_MFG_18=OPR_MFG[['Emp Id','2018']]
  OPR_MFG_18.rename(columns={"2018":"9-Box 18","Emp Id":"Employee_ID"},inplace=True)
  OPR_PROC=OPR_PROC.dropna()
  OPR_PROC_17=OPR_PROC[['Emp ID','9 Box - 2017']]
  OPR_PROC_17.rename(columns={"9 Box - 2017":"9-Box 17","Emp ID":"Employee_ID"},inplace=True)
  OPR_PROC_18=OPR_PROC[['Emp ID','9 Box - 2018 (June)']]
  OPR_PROC_18.rename(columns={"9 Box - 2018 (June)":"9-Box 18","Emp ID":"Employee_ID"},inplace=True)
  OPR_17_all=OPR_17.append([OPR_HR_17,OPR_LOGI_17,OPR_MFG_17,OPR_PROC_17]).drop_duplicates()
  OPR_18_all=OPR_18.append([OPR_HR_18,OPR_LOGI_18,OPR_MFG_18,OPR_PROC_18]).drop_duplicates()
  hclv=pd.merge(hclv, OPR_17_all, left_on="Employee_ID",right_on="Employee_ID",how="left")
  hclv=pd.merge(hclv, OPR_18_all, left_on="Employee_ID",right_on="Employee_ID",how="left")

  hclv['Most_Recent_9_Box'] = np.where(hclv['Termination Year'] == 2017,hclv['9-Box 17'],hclv['9-Box 18'])
  
  #Adding Commute Distance
  ## Read in the 4 zipcodes dataframes from the pickle files
  with open('act_zip_us.pickle','rb') as f1:
    pk_act_zip_us=pickle.load(f1)
  with open('act_zip_ca.pickle','rb') as f2:
    pk_act_zip_ca=pickle.load(f2)
  with open('term_zip_us.pickle','rb') as f3:
    pk_term_zip_us=pickle.load(f3)
  with open('term_zip_ca.pickle','rb') as f:
    pk_term_zip_ca=pickle.load(f)
  ## Merge all 4 Df into 1
  df_zip = pk_act_zip_us.append(pk_act_zip_ca).append(pk_term_zip_us).append(pk_term_zip_ca)
  df_zip = df_zip.drop_duplicates()
  df_zip = df_zip[['Employee_ID','distance']].reset_index().drop("index",axis=1)
  df_zip.rename(columns={"distance":"Office_Commute_Distance"},inplace=True)
  hclv=pd.merge(hclv,df_zip,how='left',on='Employee_ID')
  hclv.Office_Commute_Distance = pd.to_numeric(hclv.Office_Commute_Distance)
  
  #Dropping Duplicates created by OPR + Distance Join
  hclv = hclv.drop_duplicates(subset=['Employee_ID'],keep='last')
  
  #Creating Days_to_Churn 
  hclv['Days_to_Churn'] = hclv['Termination Date'] - hclv['Term_Group']
 
  #os.chdir("/home/cdsw")
  
  #from merit import create_merit
  #merit = create_merit()
  #hclv = pd.merge(hclv, merit,how='left',left_on = "Employee_ID", right_on= "merit_Employee_ID")

  #hclv = hclv.drop(['merit_Employee_ID', 'merit_Effective_Date'], axis=1)
  

  #- ECONOMIC INDICATORS 

  hclv["Adjusted Effective Date"] = pd.to_datetime(hclv['Effective Date for Current Position_x'])
  hclv["Adjusted Effective Date"] = np.where(hclv["Adjusted Effective Date"] < "2016-01-01","2016-01-01",hclv['Effective Date for Current Position_x'])

  os.chdir("/home/cdsw")
  from economic_indicators import Create_Macro
  Full_Macro = Create_Macro(0)

  #Make Effective Month and Year Columns
  hclv['Effective Date'] = pd.to_datetime(hclv['Adjusted Effective Date'])
  hclv['Effective Month'] = hclv['Effective Date'].dt.month
  hclv['Effective Year'] = hclv['Effective Date'].dt.year


  #Merge with Macro for Effective
  hclv = hclv.merge(Full_Macro, how='left', left_on=['Effective Month', 'Effective Year'], right_on=['Month','Year'])
  hclv.rename(columns={'CPI_Urban_CSL_Shifted':'CPI_Urban_CSL_Effective', 'Unemployment_Rate_Shifted':'Unemployment_Rate_Effective', 'PPI_Finished_Goods_Shifted':'PPI_Finished_Goods_Effective', 'CCI_Shifted':'CCI_Effective',
                        'Average_Hourly_Earnings_Shifted':'Average_Hourly_Earnings_Effective', 'Real_GDP_Quarterly_Shifted':'Real_GDP_Quarterly_Effective'},inplace=True)

  #Remove Time Based Columns
  hclv = hclv.drop(['Effective Month','Effective Year'], axis = 1)


  #Drop Additional Columns
  hclv = hclv.drop(['Real_GDP_Quarterly_Effective', "Adjusted Effective Date",'Effective Date','Average_Hourly_Earnings_Effective','CPI_Urban_CSL_Effective',
                    'PPI_Finished_Goods_Effective','Unemployment_Rate_Effective'], axis = 1)
  
  
  # Return completed hclv Dataframe
  return hclv
  
  







