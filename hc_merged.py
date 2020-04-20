"""
Please make sure the csv files are in the directory paths below before running this:
-/home/cdsw/Data/Headcount Data New/2017
-/home/cdsw/Data/Headcount Data New/2018
-/home/cdsw/Data/Headcount Data New/2019
-/home/cdsw/Data/Leavers Report New.xlsx
"""

### ----------- Reading in Individual **NEW** Monthly Headcount Reports and Appedning to One DataFrame
### ----------- Last Updated - Nov 15, 2019 by Monu

import sys
import csv
import glob
import pandas as pd

path_2017 = '/home/cdsw/Data/Headcount Data New/2017'
path_2018 = '/home/cdsw/Data/Headcount Data New/2018'
path_2019 = '/home/cdsw/Data/Headcount Data New/2019'


filenames_2017 = glob.glob(path_2017 + "/*.xlsx")
filenames_2018 = glob.glob(path_2018 + "/*.xlsx")
filenames_2019 = glob.glob(path_2019 + "/*.xlsx")

filenames = filenames_2017 + filenames_2018 + filenames_2019

df = pd.DataFrame()

for f in filenames:
  data = pd.read_excel(f,'Sheet1')
  df = df.append(data)
  
#Creating Final CSV Merged - ''merged_headcount_V4.csv'
df.to_csv('merged_headcount_V4.csv',index=False)
  
### ----------- Cleaning and Keeping only White Collar + Regular 

df.head()
df.shape
df.columns

df['Position Worker Type'].value_counts()
df['Job Category'].value_counts()

position_codes_to_keep = ['Regular','MBA','Trainee']

df_whr = df[(df['Position Worker Type'].isin(position_codes_to_keep)) & (df['Job Category'] == 'White Collar')]
df_whr.shape
df_whr['Report Date'].value_counts()

#Creating Final CSV Merged - 'merged_headcount_WHR_V3_new.csv'
df_whr.to_csv('merged_headcount_WHR_V4.csv',index=False)


### ----------- Leaver Report 

lv = pd.read_excel('Leavers Report New.xlsx')

lv['Job Category'].value_counts()
lv['Employee Type'].value_counts()

lv_whr = lv[(lv['Employee Type'] != '') & (lv['Job Category'] == 'White Collar')]

lv_whr.drop_duplicates().shape

lv_whr['Termination Month'] = lv_whr['Termination Date'].dt.month

lv_whr['Termination Year'] = lv_whr['Termination Date'].dt.year

lv_whr.drop_duplicates().shape

lv_whr.to_csv('leavers_WHR_V4.csv',index=False)




