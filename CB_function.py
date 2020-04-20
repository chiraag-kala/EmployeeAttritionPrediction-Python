import pandas as pd
import numpy as np
from datetime import date
import warnings
from CB_cleaning import data_processing
from hclv import create_hclv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

class CB_Recommender():
  
  def __init__(self):
    
    self._hclv, self._cosine_sim, self._cosine_sim_df,self._indices, self._df = self._get_data_()
    
    # Append additional information for input IDs
    self._info_columns = ['Employee_ID','Title', 'Churn', 'Termination Date','Age_Bucket',
                          'BUCluster_x','Compa Ratio','Most_Recent_MBO','Division_x','Job_Band_x', 
                          'Location_x', 'NewBusinessUnit_x','Years of Service', 'Most_Recent_9_Box']
    
    
  def _get_data_(self):    
    CB_frame, original_hclv = data_processing(snap_date = date.today())

    hclv = create_hclv('2090-01-01')

    # Drop people who left involuntarily
    original_hclv = original_hclv[original_hclv.Term_Reason_Final != 'Involuntary'] 

    df=CB_frame[['Employee_ID','combined']]
    df.set_index("Employee_ID", inplace=True)

    #----Build content-based recommender--------#

    tf_idf = TfidfVectorizer()
    tf_idf_matrix = tf_idf.fit_transform(df.combined)

    #----View Employees vs. Counts -------
    tokens = tf_idf.get_feature_names()

    def wm2df(wm,feat_names):
      import pandas as pd
      doc_names = ['Employee {:d}'.format(idx) for idx,_ in enumerate(wm)]
      df = pd.DataFrame(data=wm.toarray(),index=doc_names,columns=feat_names)
      return df

    tfidf_df=wm2df(tf_idf_matrix,tokens)

    #-----Append numerical columns to the df above

    cols_to_append = ['merit_Compa_Change','merit_Title_Count','merit_Bonus_Change','merit_VariableComp_Count',
                     'Unique_Level_Changes','Function_Count','Band_Count','Years of Service','BU_Count','Division_Count',
                     'merit_Base Pay Change – Percent','Most_Recent_MBO','Compa Ratio','merit_Base_Count','merit_Promo_Count',
                     'Office_Commute_Distance','merit_Band_Count','Position_Count','merit_BasePay_Count']

    num_cols = original_hclv[cols_to_append]
    num_cols.index = tfidf_df.index

    #-----Scale the numerical columns so that they are all between [0,1]
    scaler = MinMaxScaler()
    num_cols[cols_to_append] = scaler.fit_transform(num_cols[cols_to_append])
    tfidf_df = pd.concat([tfidf_df,num_cols],axis=1)


    #----Convert df back to a matrix -------
    tf_idf_matrix = tfidf_df.to_numpy()

    indices = pd.Series(df.index)
    indices[:5]

    # generate the cosine similarity matrix
    cosine_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

    # append employee IDs to rows and columns for easier reference
    cosine_sim_df = pd.DataFrame(cosine_sim, columns = indices, index = indices)
    
    return hclv, cosine_sim, cosine_sim_df,indices, df
  
  def _get_input_id_table_(self, input_id, num_rec):
    Input_ID_table = self._hclv[self._hclv.Employee_ID == input_id][self._info_columns]
    Input_ID_table = Input_ID_table[['Employee_ID','Title', 'Churn', 'Termination Date','Age_Bucket','BUCluster_x','Compa Ratio','Most_Recent_MBO',
                'Division_x','Job_Band_x', 'Location_x', 'NewBusinessUnit_x',
               'Years of Service', 'Most_Recent_9_Box']]   #This makes sure the columns are shown in the desired order
    return Input_ID_table.reset_index()
  
  
  def _get_output_id_table_(self, Input_ID, Num_Rec):
    Output_ID = self._get_recommendations_(Input_ID, Num_Rec, self._cosine_sim)
    In_Out_Sim = self._cosine_sim_df.loc[Input_ID][Output_ID].reset_index()
    In_Out_Sim.columns = ['Employee_ID','Similarity Score']
    Out_ID_table = self._hclv[self._hclv.Employee_ID.isin(Output_ID)][self._info_columns]
    Out_ID_table = pd.merge(left = Out_ID_table, right = In_Out_Sim, on = "Employee_ID", how = "left")
    Out_ID_table = Out_ID_table.sort_values('Similarity Score', ascending = False)
    Out_ID_table = Out_ID_table[Out_ID_table.Churn == 0]
    Out_ID_table = Out_ID_table[['Employee_ID','Similarity Score', 'Title', 'Age_Bucket','BUCluster_x','Compa Ratio','Most_Recent_MBO',
                    'Division_x','Job_Band_x', 'Location_x', 'NewBusinessUnit_x',
                   'Years of Service', 'Most_Recent_9_Box']]
    return Out_ID_table.reset_index()

# function that takes in employee ID as input and returns the top N recommended employees
# that are most similar to the input employee
  def _get_recommendations_(self, Employee_ID, num, cosine_sim):
    recommended_employee = []
    idx = self._indices[self._indices == Employee_ID].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_indexes = list(score_series.iloc[1:num+1].index)
    for i in top_indexes:
      recommended_employee.append(list(self._df.index)[i])

    return recommended_employee
  
    
  def _rec_generator_(self, input_id, num_rec):
    input_id_table = self._get_input_id_table_(input_id, num_rec)
    output_id_table = self._get_output_id_table_(input_id, num_rec)
    print('Generate Input Info ......\n\n')
    print(input_id_table)
    print('**************************\n\n')
    print('Genearte Output Info ......\n\n')
    print(output_id_table)
    






