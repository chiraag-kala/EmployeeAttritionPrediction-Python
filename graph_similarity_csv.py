cd

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

date_list = list((pd.date_range('2017-02-01','2019-10-01',freq = '1M') - pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d'))
for snap_date in date_list:
  from CB_cleaning_no_manager import data_processing
  CB_frame, original_hclv = data_processing(snap_date)

  from hclv import create_hclv
  hclv = create_hclv('2090-01-01')

  # Drop people who left involuntarily
  original_hclv = original_hclv[original_hclv.Term_Reason_Final != 'Involuntary'] 

  df=CB_frame[['Employee_ID','combined']]
  df.set_index("Employee_ID", inplace=True)

  #----Build content-based recommender--------#
  from sklearn.metrics.pairwise import cosine_similarity
  from sklearn.feature_extraction.text import TfidfVectorizer

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
                   'Promotion_Count','Function_Count','Band_Count','Years of Service','BU_Count','Division_Count',
                   'merit_Base Pay Change – Percent','Most_Recent_MBO','Compa Ratio','merit_Base_Count','merit_Promo_Count',
                   'Office_Commute_Distance','merit_Band_Count','Position_Count','merit_BasePay_Count']

  num_cols = original_hclv[cols_to_append]
  num_cols.index = tfidf_df.index

  #-----Scale the numerical columns so that they are all between [0,1]
  from sklearn.preprocessing import MinMaxScaler

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
  
  ### FINAL CSV HERE ###
  # save the similarity dataframe as csv
  cosine_sim_df.to_csv('cosine_similarity_v2.csv')
  
  # to check the distribution of similarity scores
  # upper_triangle = pd.DataFrame(list(cosine_sim[np.triu_indices(6694,k=1)]))
  
  #import seaborn as sns
  #sns.distplot(upper_triangle)
