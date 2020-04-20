#cd /home/cdsw

import pandas as pd
import numpy as np
import pickle
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import os

### TICKER ##################################################################################
  

#cd /home/cdsw/Data

os.chdir("/home/cdsw/Data")
  
def Create_Ticker(days_shifted):

  #Import
  #os.chdir("/home/cdsw/Data")
  import pandas as pd
  import numpy as np
  import sys
  import re
  import os
  
  os.chdir("/home/cdsw/Data")
  Ticker = pd.read_csv('Ticker.csv')
  df = Ticker
  
  #Shift
  Shifted = Ticker.set_index(['Date']).shift(days_shifted) [['S&P_Open', 'S&P_High',
                                                'S&P_Low', 'S&P_Close', 
                                                'S&P_Adj Close','S&P_Volume', 
                                                ' KHC_Close/Last', ' KHC_Volume',
                                                ' KHC_Open',' KHC_High', ' KHC_Low']]
  #Shift Ticker current ticker to Last Month
  New_Ticker = Ticker.set_index(['Date']).shift(30) [['S&P_Open', 'S&P_High',
                                                'S&P_Low', 'S&P_Close', 
                                                'S&P_Adj Close','S&P_Volume', 
                                                ' KHC_Close/Last', ' KHC_Volume',
                                                ' KHC_Open',' KHC_High', ' KHC_Low']]
  
  #Rename Shifted Columns
  Shifted = Shifted.rename(columns = {'S&P_Open': 'S&P_Open_Shifted', 'S&P_High':'S&P_High_Shifted',
                                                'S&P_Low':'S&P_Low_Shifted', 'S&P_Close':'S&P_Close_Shifted', 
                                                'S&P_Adj Close':'S&P_Adj Close_Shifted',
                                                'S&P_Volume':'S&P_Volume_Shifted',' KHC_Close/Last':'KHC_Close/Last_Shifted',
                                                ' KHC_Volume':'KHC_Volume_Shifted',' KHC_Open':'KHC_Open_Shifted',
                                                ' KHC_High':'KHC_High_Shifted', ' KHC_Low':'KHC_Low_Shifted'})
  #Join Original with Shifted Columns
  Ticker = New_Ticker.merge(Ticker[['Year', 'Month','Date']], how = 'left',on='Date')
  Full_Ticker = Ticker.merge(Shifted, how='left', on='Date')
  
  #Clean $ signs an convert to float
  Full_Ticker[' KHC_Close/Last'] = Full_Ticker[' KHC_Close/Last'].str.replace('$','').astype('float')
  Full_Ticker[' KHC_Open'] = Full_Ticker[' KHC_Open'].str.replace('$','').astype('float')
  Full_Ticker[' KHC_High'] = Full_Ticker[' KHC_High'].str.replace('$','').astype('float')
  Full_Ticker[' KHC_Low'] = Full_Ticker[' KHC_Low'].str.replace('$','').astype('float')

  Full_Ticker['KHC_Close/Last_Shifted'] = Full_Ticker['KHC_Close/Last_Shifted'].str.replace('$','').astype('float')
  Full_Ticker['KHC_Open_Shifted'] = Full_Ticker['KHC_Open_Shifted'].str.replace('$','').astype('float')
  Full_Ticker['KHC_High_Shifted'] = Full_Ticker['KHC_High_Shifted'].str.replace('$','').astype('float')
  Full_Ticker['KHC_Low_Shifted'] = Full_Ticker['KHC_Low_Shifted'].str.replace('$','').astype('float')
  
  #Group by Year and Month
  Full_Ticker = Full_Ticker.groupby(['Year', 'Month']).mean().reset_index()

  #Shortened Dataframe 
  Full_Ticker = Full_Ticker[['Year', 'Month', 'S&P_Close', 'S&P_Close_Shifted', ' KHC_Close/Last', 'KHC_Close/Last_Shifted']]

  # Calculation 
  #Full_Ticker['Relative_KHC_SP_Perfromance'] = (Full_Ticker[' KHC_Close/Last'] / Full_Ticker['S&P_Close']) - (Full_Ticker['KHC_Close/Last_Shifted'] / Full_Ticker['S&P_Close_Shifted'])
  
  ##Join to Add Date
  #Full_Ticker = Full_Ticker.merge(df[['Date','Month','Year']], how='left', left_on=['Month','Year'], right_on=['Month','Year'])
  
  #Change Directory Back
  os.chdir("/home/cdsw")
  
  #Return df
  return Full_Ticker


#Try Output
#Full_Ticker = Create_Ticker(365)



### Macro ##################################################################################

#cd /home/cdsw/Data


def Create_Macro(days_shifted):
  
  #Import
  import pandas as pd
  import numpy as np
  import sys
  import re
  import os
  os.chdir("/home/cdsw/Data")
  Macro = pd.read_csv('Macro Economic Indicators.csv', encoding ='ISO-8859-1')
  df = Macro
  
  ##Rename
  Macro = Macro.rename(columns={'DATE':'Date', 'CPI_Urban_CSL': 'CPI_Urban_CSL', df.columns[2]:'Unemployment_R', 'PPI_Finished_Goods':'PPI_Finished_Goods',
       'CCI':'CCI', 'Average_Hourly_Earnings':'Average_Hourly_Earnings', 'Real_GDP_Quarterly':'Real_GDP_Quarterly', 'Month':'Month',
       'Year':'Year'})

  
  #Shift
  Shifted = Macro.set_index(['Date']).shift(days_shifted)[['CPI_Urban_CSL', 'Unemployment_R', 'PPI_Finished_Goods',
                                                           'CCI', 'Average_Hourly_Earnings', 'Real_GDP_Quarterly']]
  #Rename Shifted Columns
  Shifted = Shifted.rename(columns = {'CPI_Urban_CSL':'CPI_Urban_CSL_Shifted', 'Unemployment_R':'Unemployment_Rate_Shifted', 
                                      'PPI_Finished_Goods':'PPI_Finished_Goods_Shifted','CCI':'CCI_Shifted', 
                                      'Average_Hourly_Earnings':'Average_Hourly_Earnings_Shifted', 
                                      'Real_GDP_Quarterly':'Real_GDP_Quarterly_Shifted'})
  
  #Join Original with Shifted Columns
  Full_Macro = Macro.merge(Shifted, how='left', on='Date')
  
  #Group by Year and Month
  Full_Macro = Full_Macro.groupby(['Year', 'Month']).mean().reset_index()
  
  #Column Selection
  Full_Macro = Full_Macro[['CPI_Urban_CSL_Shifted', 'Unemployment_Rate_Shifted', 'PPI_Finished_Goods_Shifted', 'CCI_Shifted',
                          'Average_Hourly_Earnings_Shifted', 'Real_GDP_Quarterly_Shifted', 'Month','Year']]

  #Change Directory Back
  os.chdir("/home/cdsw")
  
  #Return df
  return Full_Macro


##Try Output:
#Full_Macro = Create_Macro(100)


##### MERGE HCLV with TICKER AND MACRO ##################################################################################
#from hclv import create_hclv
#hclv = create_hclv()

#Make Month and Year Columns
#from datetime import datetime
#hclv['Report Date'] = pd.to_datetime(hclv['Report Date'])
#hclv['Report Month'] = hclv['Report Date'].dt.month
#hclv['Report Year'] = hclv['Report Date'].dt.year

#Merge with Ticker
#hclv = hclv.merge(Full_Ticker, how='left', left_on=['Report Month', 'Report Year'], right_on=['Month','Year'])

#Merge with Macro
#hclv = hclv.merge(Full_Macro, how='left', left_on=['Report Month', 'Report Year'], right_on=['Month','Year'])








