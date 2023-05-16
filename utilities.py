# Author: Yeon Mi Hwang, Qi Wei 
# Date: 5/15/23

from datetime import date, datetime, timedelta
from dateutil.relativedelta import *
import pyspark.sql.functions as F
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import sklearn 
%matplotlib inline
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import shuffle
import math


import CEDA_tools_ver20221220.load_ceda_etl_tools


def impute_missing_data(df):
  df.fillna(value=pd.np.nan, inplace=True)
  imputer = SimpleImputer(missing_values=np.nan, strategy='median')
  imputer.fit_transform(df)
  return(imputer.fit_transform(df))


def get_matching_pairs(df_experimental, df_control, scaler=True):
  if scaler:
      scaler = StandardScaler()
      scaler.fit(df_experimental.append(df_control))
      df_experimental_scaler = scaler.transform(df_experimental)
      df_control_scaler = scaler.transform(df_control)
      nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(df_control_scaler)
      distances, indices = nbrs.kneighbors(df_experimental_scaler)
  
  else:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean').fit(df_control)
    distances, indices = nbrs.kneighbors(df_experimental)
  indices = indices.reshape(indices.shape[0])
  matched = df_control.iloc[indices, :]
  return matched

def select_psm_columns(df, columns):
  return_df = df.select(*columns)
  return return_df


def consolidate_race_responses(l):
  l_new = []
  for i in l:
    if i == 'White':
      l_new.append('White or Caucasian')
    elif i == 'Patient Refused' or i == 'Unable to Determine' or i == 'Declined' or i == 'Unknown':
      continue
    else:
      l_new.append(i)
  l_new = list(set(l_new))
  return(l_new)


def handle_multiracial_exceptions(l):
  l_new = consolidate_race_responses(l)
  if l_new is None:
    return('Unknown')
  if len(l_new) == 1:
    return(l_new[0])
  if 'Other' in l_new:
    l_new.remove('Other')
    if l_new is None:
      return('Other')
    if len(l_new) == 1:
      return(l_new[0])
  return('Multiracial')


def format_race(i):
  if i is None:
    return('Unknown')
  if len(i) > 1:
    return('Multiracial')
  if i[0] == 'White':
    return('White or Caucasian')
  if i[0] == 'Declined' or i[0] == 'Patient Refused':
    return('Unknown')
  if i[0] == 'Unable to Determine':
    return('Unknown')
  return(i[0])


def format_ethnic_group(i):
  if i is None:
    return 'Unknown'
  if i == 'American' or i == 'Samoan':
    return 'Not Hispanic or Latino'
  elif i == 'Filipino' or i == 'Hmong':
    return 'Not Hispanic or Latino'
  elif i == 'Sudanese':
    return 'Not Hispanic or Latino'
  if i == 'Patient Refused' or i == 'None':
    return 'Unknown'
  return i


def format_parity(i):
  if i is None:
    return 0
  i = int(i)
  if i == 0 or i == 1:
    return 0
  if i > 1 and i < 5:
    return 1
  if i >= 5:
    return 2
  return 0


def format_gravidity(gravidity):
  if gravidity is None:
    return 0
  gravidity = int(gravidity)
  if gravidity == 0 or gravidity == 1:
    return 0
  elif gravidity > 1 and gravidity < 6:
    return 1
  elif gravidity >= 6:
    return 2
  return 0
    
  
def format_preterm_history(preterm_history, gestational_days):

  if preterm_history is None:
    return 0
  else:
    preterm_history = int(preterm_history)
    if preterm_history == 0 or (preterm_history == 1 and gestational_days < 259):
      return 0
    else:
      return 1
  return 0


def encode_delivery_method(i):
  '''
  0 = Vaginal
  1 = C-Section
  -1 = Unknown
  '''
  list_vaginal = ['Vaginal, Spontaneous',
       'Vaginal, Vacuum (Extractor)',
       'Vaginal, Forceps', 'Vaginal < 20 weeks',
       'Vaginal, Breech', 'VBAC, Spontaneous',
       'Vaginal Birth after Cesarean Section',
       'Spontaneous Abortion']
  list_c_section = ['C-Section, Low Transverse',
       'C-Section, Low Vertical',
       'C-Section, Classical',
       'C-Section, Unspecified']
  if i in list_vaginal:
    return(0)
  if i in list_c_section:
    return(1)
  return(-1)


def encode_bmi(bmi):
  if bmi is None or math.isnan(bmi):
    return -1
  bmi = int(bmi)
  if bmi >= 15 and bmi < 18.5:
    return 0
  if bmi < 25:
    return 1
  if bmi < 30:
    return 2
  if bmi < 35:
    return 3
  if bmi < 40:
    return 4
  return -1

def encode_ruca(ruca):
  if ruca is None:
    return -1
  if ruca == 'Rural':
    return 0
  if ruca == 'SmallTown':
    return 1
  if ruca == 'Micropolitan':
    return 2
  if ruca == 'Metropolitan':
    return 3
  return -1

def encode_age(age):
  if age < 25:
    return 0
  if age < 30:
    return 1
  if age < 35:
    return 2
  if age < 40:
    return 3
  if age < 45:
    return 4
  return -1




def handle_missing_bmi(df):
  print('# Percent of patients with pregravid BMI:', str(round(100*(len(df) - df['pregravid_bmi'].isna().sum())/len(df), 1)), '%')
  print('Imputing median pregravid BMI of', str(round(df['pregravid_bmi'].median(), 2)), '...')
  df['pregravid_bmi'].fillna(df['pregravid_bmi'].median(), inplace = True)
  print('\n')
  return df

def handle_missing_svi(df, col):
  print('# Percent of patients with svi:', str(round(100*(len(df) - df[col].isna().sum())/len(df), 1)), '%')
  print('Imputing median svi of', str(round(df[col].median(), 2)), '...')
  df[col].fillna(df[col].median(), inplace = True)
  print('\n')
  return df

from sklearn.preprocessing import MinMaxScaler

def format_dataframe_for_psm(df, select_columns):
  dict_white = {'White or Caucasian': 1, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_asian = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 1, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_multiracial = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 1, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_other = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 1, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_black = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 1, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 0}
  dict_pacific_islander = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 1, 'American Indian or Alaska Native': 0}
  dict_native_american = {'White or Caucasian': 0, 'Unknown': 0, 'Asian': 0, 'Multiracial': 0, 'Other': 0, 'Black or African American': 0, 'Native Hawaiian or Other Pacific Islander': 0, 'American Indian or Alaska Native': 1}
  dict_ethnic_groups = {None:-1, 'Unknown_NotReported': -1, 'Hispanic_Latino': 1, 'Not_Hispanic_Latino': 0}
  dict_fetal_sex = {None: -1, 'Male': 1, 'Female': 0, 'Other': -1, 'Unknown': -1}
  dict_commercial_insurance = {'Medicaid': 0, 'Medicare': 0, 'Uninsured-Self-Pay': 0, None: 0, 'Other': 0, 'Commercial': 1}
  dict_governmental_insurance = {'Medicaid': 1, 'Medicare': 0, 'Uninsured-Self-Pay': 0, None: 0, 'Other': 0, 'Commercial': 0}
  dict_wildtype = {'wild_type' : 1, 'alpha' : 0, 'delta' : 0, 'omicron1' : 0, 'omicron2': 0, 'omicron': 0}
  dict_alpha = {'wild_type' : 0, 'alpha' : 1, 'delta' : 0, 'omicron1' : 0, 'omicron2': 0, 'omicron': 0}
  dict_delta = {'wild_type' : 0, 'alpha' : 0, 'delta' : 1, 'omicron1' : 0, 'omicron2': 0, 'omicron': 0}
  dict_omicron = {'wild_type' : 0, 'alpha' : 0, 'delta' : 0, 'omicron1' : 1, 'omicron2': 1, 'omicron': 1}
  dict_covid_1st = {'1st trimester' : 1, '2nd trimester' : 0, '3rd trimester' : 0}
  dict_covid_2nd = {'1st trimester' : 0, '2nd trimester' : 1, '3rd trimester' : 0}
  dict_covid_3rd = {'1st trimester' : 0, '2nd trimester' : 0, '3rd trimester' : 1}
  rest_columns = [i for i in df.columns if i not in select_columns]
  keep_columns = ['gestational_days', 'insurance', 'race', 'lmp', 'delivery_year']
  intact_df = df.select(*(rest_columns+keep_columns))
  selected_pd = select_psm_columns(df, select_columns).toPandas()
  selected_pd = pd.get_dummies(selected_pd, columns = ['delivery_year'], prefix = 'year')
  for index, row in selected_pd.iterrows():
    selected_pd.at[index, 'race'] = format_race(row['race'])
    selected_pd.at[index, 'Preterm_history'] = format_preterm_history(row['Preterm_history'], row['gestational_days'])
  for index, row in selected_pd.iterrows():
    selected_pd.at[index, 'race_white'] = dict_white[row['race']]
    selected_pd.at[index, 'race_asian'] = dict_asian[row['race']]
    selected_pd.at[index, 'race_black'] = dict_black[row['race']]
    selected_pd.at[index, 'race_other'] = dict_other[row['race']]
    selected_pd.at[index, 'ethnic_group'] = dict_ethnic_groups[row['ethnic_group']]
    selected_pd.at[index, 'ob_hx_infant_sex'] = dict_fetal_sex[row['ob_hx_infant_sex']]
    selected_pd.at[index, 'commercial_insurance'] = dict_commercial_insurance[row['insurance']]
    selected_pd.at[index, 'Parity'] = format_parity(row['Parity'])
    selected_pd.at[index, 'Gravidity'] = format_gravidity(row['Gravidity'])
    selected_pd.at[index, 'pregravid_bmi'] = encode_bmi(row['pregravid_bmi'])
    selected_pd.at[index, 'age_at_start_dt'] = encode_age(row['age_at_start_dt'])
    selected_pd.at[index, 'ruca_categorization'] = encode_ruca(row['ruca_categorization'])
  selected_pd = selected_pd.drop(columns=['gestational_days', 'insurance', 'race', 'lmp'])
  selected_pd = handle_missing_svi(selected_pd, 'RPL_THEME1')
  selected_pd = handle_missing_svi(selected_pd, 'RPL_THEME2')
  selected_pd = handle_missing_svi(selected_pd, 'RPL_THEME3')
  selected_pd = handle_missing_svi(selected_pd, 'RPL_THEME4' )
  selected_pd = selected_pd.fillna(-1)
  print('Columns used for matching:')
  for col in selected_pd.columns:
    print(col)
  print('\n')
  print('\n')
  intact_pd = intact_df.toPandas()
  final_pd = pd.concat([intact_pd, selected_pd], axis=1)
  return(final_pd, selected_pd.columns)



def get_propensity_score(T_col, X_cols, df):
  from sklearn.linear_model import LogisticRegression
  ps_model = LogisticRegression(C=1e6, max_iter=10000, solver='lbfgs').fit(df[X_cols], df[T_col])
  data_ps = df.assign(propensity_score=ps_model.predict_proba(df[X_cols])[:, 1])
  return (data_ps)

def propensity_score_matching(df, x):
  treatment = df[x] 
  mask = treatment == 1
  pscore = df['propensity_score']
  pos_pscore = np.asarray(pscore[mask])
  neg_pscore = np.asarray(pscore[~mask])
  print('treatment count:', pos_pscore.shape)
  print('control count:', neg_pscore.shape)
  from sklearn.neighbors import NearestNeighbors
  if len(neg_pscore) > len(pos_pscore):
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(neg_pscore.reshape(-1, 1))
    distances, indices = knn.kneighbors(pos_pscore.reshape(-1, 1))
    df_pos = df[mask]
    df_neg = df[~mask].iloc[indices[:, 0]]
  else: 
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=1)
    knn.fit(pos_pscore.reshape(-1, 1))
    distances, indices = knn.kneighbors(neg_pscore.reshape(-1, 1))
    df_pos = df[mask].iloc[indices[:, 0]] 
    df_neg = df[~mask]
  df_matched = pd.concat([df_pos, df_neg], axis=0)
  print (df_matched.shape)
  return (df_matched )


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


 def add_geo_features(cohort_df, geo_df_name, join_cols = ['pat_id', 'instance']):
  geodf_list = ['ruca2010revised', 'countytypologycodes2015', 'farcodeszip2010', 'ruralurbancontinuumcodes2013', 'urbaninfluencecodes2013', 'svi2018_us', 'svi2018_us_county']
  master_patient = spark.sql("SELECT * FROM rdp_phi.dim_patient_master").select('pat_id','instance', 'PATIENT_STATE_CD', 'PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED', 'ZIP')
  
  if geo_df_name not in geodf_list:
    print ('incorrect geo df name')
  else:
    geo_df = spark.sql("SELECT * from rdp_phi_sandbox.{0}".format(geo_df_name))
    if geo_df_name == 'ruca2010revised':
      geo_df = geo_df.withColumn('FIPS', F.col('State_County_Tract_FIPS_Code').cast(StringType())).drop('State_County_Tract_FIPS_Code')
      master_patient = master_patient.withColumn("FIPS", F.expr("CASE WHEN PATIENT_STATE_CD = 'CA' THEN substring(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED, 2, length(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED)-2) ELSE substring(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED, 0, length(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED)-1) END"))
      joined_df = master_patient.join(geo_df, 'FIPS', 'inner')
    elif geo_df_name == 'svi2018_us':
      master_patient = master_patient.withColumn("FIPS", F.expr("substring(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED, 0, length(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED)-1)"))
      joined_df = master_patient.join(geo_df, 'FIPS', 'inner') 
    elif ((geo_df_name == 'countytypologycodes2015')|(geo_df_name == 'urbaninfluencecodes2013')):
      geo_df = geo_df.withColumn('FIPS4', F.col('FIPStxt').cast(StringType())).drop('FIPStxt')
      master_patient = master_patient.withColumn("FIPS4", F.expr("CASE WHEN PATIENT_STATE_CD = 'CA' THEN substring(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED, 2, 4) ELSE substring(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED, 0, 5) END"))
      joined_df = master_patient.join(geo_df, 'FIPS4', 'inner')
    elif ((geo_df_name == 'svi2018_us_county')|(geo_df_name == 'ruralurbancontinuumcodes2013')):
      geo_df = geo_df.withColumn('FIPS5', F.col('FIPS').cast(StringType()))
      master_patient = master_patient.withColumn("FIPS5", F.expr("substring(PATIENT_ADDR_CENSUS_BLOCKGROUP_DERIVED, 0, 5)"))
      joined_df = master_patient.join(geo_df, 'FIPS5', 'inner')    
    elif geo_df_name == 'farcodeszip2010':
      geo_df = geo_df.withColumn('ZIP5', F.col('ZIP').cast(StringType())).drop('ZIP')
      master_patient = master_patient.withColumn("ZIP5", F.expr("substring(ZIP, 0, 5)")).drop('ZIP')
      joined_df = master_patient.join(geo_df, 'ZIP5', 'inner')
    return_df = cohort_df.join(joined_df, join_cols, 'left')
  return return_df 

def categorize_ruca(code):
  if code is None:
    return None
  elif code < 4:
    return 'Metropolitan'
  elif code < 7:
    return 'Micropolitan'
  elif code < 10:
    return 'SmallTown'
  elif code < 99:
    return 'Rural'
  elif code == 99:
    return 'NotCoded'
  else:
    return None 
categorize_ruca_udf = F.udf(lambda code: categorize_ruca(code), StringType())


def get_count(df):
  ptb_count = 0
  sga_count = 0
  lbw_count = 0
  csec_count = 0
  for index, row in df.iterrows():
    ptb = row['preterm_category']
    sga = row['SGA']
    lbw = row['LBW']
    csec = row['delivery_method']
    if ptb == 'preterm':
      ptb_count += 1
    if lbw == 'LBW':
      lbw_count += 1
    if sga == 'SGA':
      sga_count += 1
    if csec == 'c-section':
      csec_count += 1
  return ptb_count, sga_count, lbw_count, csec_count


def get_pval_signal (p):
  if p>0.1:
    return 'ns'
  elif p>0.05:
    return '+'
  elif p>0.01:
    return '*'
  elif p>0.001:
    return '**'
  elif p>0.0001:
    return '***'
  else:
    return '****'

 

def get_exposure_period(exposure_date):
  if exposure_date < -180:
    return -1
  elif -180 <= exposure_date < 0:
    return 0
  elif 0 <= exposure_date < 84:
    return 1 
  elif 84 <= exposure_date < 189:
    return 2
  elif 189 <= exposure_date <= 320:
    return 3
  elif 320 < exposure_date:
    return 4
  else:
    return None 
get_exposure_period_udf = F.udf(lambda exp: get_exposure_period(exp), StringType())

def get_exposure0_duration(orderstart_lmp, orderend_lmp):
  if (orderstart_lmp < -180) and (orderend_lmp < -180):
    return 0
  elif (orderstart_lmp < -180) and (-180 <= orderend_lmp < 0):
    return orderend_lmp - 180 + 1
  elif (orderstart_lmp < -180) and (0 <= orderend_lmp):
    return 180
  elif (-180 <= orderstart_lmp < 0) and  (-180 <= orderend_lmp < 0):
    return orderend_lmp - orderstart_lmp + 1
  elif (-180 <= orderstart_lmp < 0) and (0 <= orderend_lmp):
    return 0 - orderstart_lmp + 1
  elif (0 <= orderstart_lmp) and (0 <= orderend_lmp):
    return 0
  else:
    return None
def get_exposure1_duration(orderstart_lmp, orderend_lmp):
  if ( orderstart_lmp < 0) and ( orderend_lmp < 0):
    return 0
  elif ( orderstart_lmp < 0) and (0 <= orderend_lmp < 84):
    return orderend_lmp + 1 
  elif ( orderstart_lmp < 0) and (84 <= orderend_lmp ):
    return 84 
  elif (0 <= orderstart_lmp < 84) and (0 <= orderend_lmp < 84):
    return orderend_lmp - orderstart_lmp + 1
  elif (0 <= orderstart_lmp < 84) and (84 <= orderend_lmp):
    return 84 - orderstart_lmp + 1
  elif (84 <= orderstart_lmp) and (84 <= orderend_lmp):
    return 0
  else:
    return 0
  
def get_exposure2_duration(orderstart_lmp, orderend_lmp):
  if (orderstart_lmp < 84) and (orderend_lmp < 84):
    return 0
  elif (orderstart_lmp < 84) and (84 <= orderend_lmp < 189):
    return orderend_lmp - 84 + 1 
  elif (91 <= orderstart_lmp < 189) and (84 <= orderend_lmp < 189):
    return orderend_lmp - orderstart_lmp + 1
  elif ( orderstart_lmp < 84) and (189 <= orderend_lmp ):
    return 189-84
  elif (84 <= orderstart_lmp < 189) and (189 <= orderend_lmp):
    return 189 - orderstart_lmp + 1
  elif (189 <= orderstart_lmp) and (189 <= orderend_lmp):
    return 0
  else:
    return 0

def get_exposure3_duration(orderstart_lmp, orderend_lmp, gestational_days):
  if (orderstart_lmp < 189) and (orderend_lmp < 189):
    return 0
  elif (orderstart_lmp < 189) and (189 <= orderend_lmp <= gestational_days):
    return orderend_lmp -189 + 1 
  elif (orderstart_lmp < 189) and (189 <= gestational_days <= orderend_lmp) :
    return gestational_days -189 + 1 
  elif (189 <= orderstart_lmp <= gestational_days) and (189 <= orderend_lmp <= gestational_days):
    return orderend_lmp - orderstart_lmp + 1
  elif (189 <= orderstart_lmp <= gestational_days) and (gestational_days <= orderend_lmp):
    return gestational_days - orderstart_lmp + 1
  else:
    return 0

get_exposure0_duration_udf = F.udf(lambda start, end: get_exposure0_duration(start, end), IntegerType())
get_exposure1_duration_udf = F.udf(lambda start, end: get_exposure1_duration(start, end), IntegerType())
get_exposure2_duration_udf = F.udf(lambda start, end: get_exposure2_duration(start, end), IntegerType())
get_exposure3_duration_udf = F.udf(lambda start, end, ga: get_exposure3_duration(start, end, ga), IntegerType())

def get_medinfo_cohort(cohort_df):
  result_df = cohort_df.withColumn('orderstart_lmp', F.datediff(F.col('start_date'), F.col('lmp'))) .\
                        withColumn('orderend_lmp',  F.datediff(F.col('end_date'), F.col('lmp'))) .\
                        withColumn('order_duration', F.datediff(F.col('end_date'), F.col('start_date'))+1) .\
                        withColumn('orderstart_lmp_period', get_exposure_period_udf(F.col('orderstart_lmp'))) .\
                        withColumn('orderstart_lmp_period', get_exposure_period_udf(F.col('orderstart_lmp'))) .\
                        withColumn('exposure0_duration', get_exposure0_duration_udf(F.col('orderstart_lmp'), F.col('orderend_lmp'))) .\
                        withColumn('exposure1_duration', get_exposure1_duration_udf(F.col('orderstart_lmp'), F.col('orderend_lmp'))) .\
                        withColumn('exposure2_duration', get_exposure2_duration_udf(F.col('orderstart_lmp'), F.col('orderend_lmp'))) .\
                        withColumn('exposure3_duration', get_exposure3_duration_udf(F.col('orderstart_lmp'), F.col('orderend_lmp'), F.col('gestational_days'))) .\
                        withColumn('exposure_total_duration', (F.col('exposure1_duration')+F.col('exposure2_duration')+F.col('exposure3_duration'))) .\
                        withColumn('exposure1_status', F.when(F.col('exposure1_duration') > 0,1).otherwise(0)) .\
                        withColumn('exposure2_status', F.when(F.col('exposure2_duration') > 0,1).otherwise(0)) .\
                        withColumn('exposure3_status', F.when(F.col('exposure3_duration') > 0,1).otherwise(0)) .\
                        withColumn('exposure_total_status', F.when(F.col('exposure_total_duration') > 0,1).otherwise(0)) .\
                        withColumn('exposure0_status', F.when(F.col('exposure0_duration') > 0,1).otherwise(0))
  return result_df



 # below is written by Qi Wei 
 ##################################################################
## Previous omop table versions:
## 2020-08-05, 2022-02-11
## Input:
## the code stored in the dictionary based on the current keyword
## Output:
## A dataframe of codes and their corresponding descendant codes
##################################################################

## Notice!! Please always confirm the version of the OMOP table you are using, check the data panel to see if there is a newer version

print("Please always confirm the version of the OMOP table you are using, check the data panel to see if there is a newer version!!!")

def get_all_descendant_snomed_codes(code, omop_table_version):
  descendant_snomed_codes_df = spark.sql(
  """
  SELECT
    oc1.concept_code as ancestor_snomed_code,
    oc1.concept_name as ancestor_concept_name,
    oc2.concept_code as descendant_snomed_code,
    oc2.concept_name as descendant_concept_name
  FROM (
    SELECT * FROM rdp_phi_sandbox.omop_concept_{version}
    WHERE
      concept_code = {snomed_code} AND
      vocabulary_id = 'SNOMED') as oc1
  JOIN rdp_phi_sandbox.omop_concept_ancestor_{version} as oca
  ON oc1.concept_id = oca.ancestor_concept_id
  JOIN rdp_phi_sandbox.omop_concept_{version} as oc2
  ON oca.descendant_concept_id = oc2.concept_id
  ORDER BY min_levels_of_separation, oc2.concept_name
  """.format(snomed_code=code, version = omop_table_version))
  return descendant_snomed_codes_df


  ####################################################################################################################################################
## Get needed cols from problemlist table and fill null values in noted_date with closest date_of_entry info
problemlist_df = spark.sql(""" select distinct pat_id, dx_id, instance, NOTED_DATE, RESOLVED_DATE, DATE_OF_ENTRY from rdp_phi.problemlist""")

## Use date_of_entry to fill in as many null rows as possible in the noted_date column
## The logic is to use date of entry to fill in noted date, if resolved date is also unknown or >= date of entry
problemlist_df = problemlist_df.withColumn('new_noted_date', when( (problemlist_df.NOTED_DATE.isNull() & (problemlist_df.RESOLVED_DATE.isNull() | (problemlist_df.RESOLVED_DATE >= problemlist_df.DATE_OF_ENTRY) ) ), col('DATE_OF_ENTRY') )\
                                                        .otherwise(col('NOTED_DATE')) )

## Select only needed columns and then rename the column back to "noted_date" to keep following codes working
select_cols = ("pat_id", "dx_id", "instance", "RESOLVED_DATE", "new_noted_date")
problemlist_df = problemlist_df.select(*select_cols).dropDuplicates()
problemlist_df = problemlist_df.withColumnRenamed('new_noted_date', 'noted_date')

####################################################################################################################################################
## Get needed cols from encounter diagnosis table
encounterdiagnosis_df = spark.sql(""" select distinct pat_id, instance, dx_id, PAT_ENC_CSN_ID, DIAGNOSISNAME from rdp_phi.encounterdiagnosis""")

################################################################################
## full outer join to get a more detailed diagnosis table
encounterdiagnosis_df = encounterdiagnosis_df.withColumnRenamed('pat_id', 'pat_id2').withColumnRenamed('dx_id', 'dx_id2').withColumnRenamed('instance', 'instance2')
cond = [problemlist_df.pat_id == encounterdiagnosis_df.pat_id2, problemlist_df.dx_id == encounterdiagnosis_df.dx_id2, problemlist_df.instance == encounterdiagnosis_df.instance2]
prob_enc_diagnosis_df = problemlist_df.join(encounterdiagnosis_df, cond, "fullouter")

## merge pat_id2 and dx_id2 back to pat_id and dx_id columns
prob_enc_diagnosis_df = prob_enc_diagnosis_df.withColumn('pat_id_merge', when(prob_enc_diagnosis_df.pat_id.isNull(), col('pat_id2') )\
                                                        .otherwise(col('pat_id')) )\
                                              .withColumn('dx_id_merge', when(prob_enc_diagnosis_df.dx_id.isNull(), col('dx_id2') )\
                                                        .otherwise(col('dx_id')) )\
                                              .withColumn('instance_merge', when(prob_enc_diagnosis_df.instance.isNull(), col('instance2') )\
                                                        .otherwise(col('instance')) )

## Drop not needed columns
select_cols = ("pat_id_merge", "dx_id_merge", "instance_merge", "PAT_ENC_CSN_ID", "NOTED_DATE", "RESOLVED_DATE", "DIAGNOSISNAME")
prob_enc_diagnosis_df = prob_enc_diagnosis_df.select(*select_cols).dropDuplicates()

###################################################################################################
## left join with the encounter table using both pat_id and pat_enc_csn_id to get the contact_date
encounter_df = spark.sql(""" select distinct pat_id, instance, PAT_ENC_CSN_ID, CONTACT_DATE from rdp_phi.encounter""")
encounter_df = encounter_df.withColumnRenamed('pat_id', 'pat_id2').withColumnRenamed('PAT_ENC_CSN_ID', 'PAT_ENC_CSN_ID2')
cond = [prob_enc_diagnosis_df.pat_id_merge == encounter_df.pat_id2, 
        prob_enc_diagnosis_df.PAT_ENC_CSN_ID == encounter_df.PAT_ENC_CSN_ID2, 
        prob_enc_diagnosis_df.instance_merge == encounter_df.instance]

prob_enc_diag_enc_df = prob_enc_diagnosis_df.join(encounter_df, cond, "left")

## Drop not needed columns
select_cols = ("pat_id_merge", "dx_id_merge", "instance_merge", "CONTACT_DATE", "NOTED_DATE", "RESOLVED_DATE", "DIAGNOSISNAME")
prob_enc_diag_enc_df = prob_enc_diag_enc_df.select(*select_cols).dropDuplicates()

#########################################################################################################
## left join with external concept mapping table to get the corresponding snomed codes for diagnosis
externalmapping_df = spark.sql(""" select distinct value, name, instance, concept from rdp_phi.externalconceptmapping""")
cond = [prob_enc_diag_enc_df.dx_id_merge == externalmapping_df.value, 
        prob_enc_diag_enc_df.instance_merge == externalmapping_df.instance]

prob_enc_diag_enc_concept_df = prob_enc_diag_enc_df.join(externalmapping_df, cond, "left")

## Drop not needed columns
select_cols = ("pat_id_merge", "dx_id_merge", "instance_merge", "CONTACT_DATE", "NOTED_DATE", "RESOLVED_DATE", "DIAGNOSISNAME", "name", "concept")
prob_enc_diag_enc_concept_df = prob_enc_diag_enc_concept_df.select(*select_cols).dropDuplicates()

####################################################################################
## concanate the diagnosisname and name columns
df_full_diagnosis_name = prob_enc_diag_enc_concept_df.withColumn('full_diagnosis_name', when(col('diagnosisname').contains(col('name')),  col('diagnosisname'))\
                                                                 .otherwise( concat(col('diagnosisname'), lit('_'), col('name')) )
)

## Drop diagnosisname and name col
df_full_diagnosis_name = df_full_diagnosis_name.drop( *('diagnosisname', 'name') )

#####################################################################################
## Use contact_date to fill in as many null rows as possible in the noted_date column
## The logic is to: 
## 1. use window to find the earliest contact date for a combination of pat_id, dx_id, and instance
## 2. use contact date to fill in noted date, if resolved date is also unknown or >= contact_date
##
## Notes: partitionBy can take a list of cols as input; orderBy is ascending be default, so the first row will have the smallest value
partition_cols = ["pat_id_merge", "dx_id_merge", "instance_merge", "full_diagnosis_name"]
w2 = Window.partitionBy(partition_cols).orderBy(col("contact_date"))

df_full_diagnosis_name = df_full_diagnosis_name.withColumn("row",row_number().over(w2)) \
                          .filter(col("row") == 1).drop("row")

df_full_diagnosis_name = df_full_diagnosis_name.withColumn('new_noted_date', when( (df_full_diagnosis_name.NOTED_DATE.isNull() & (df_full_diagnosis_name.RESOLVED_DATE.isNull() | (df_full_diagnosis_name.RESOLVED_DATE >= df_full_diagnosis_name.CONTACT_DATE) ) ), col('CONTACT_DATE') )\
                                                        .otherwise(col('NOTED_DATE')) )

###############################################################################################
## Rename pat_id_merge, dx_id_merge, instance_merge, full_diagnosis_name for future codes
df_full_diagnosis_name = df_full_diagnosis_name.withColumnRenamed('pat_id_merge', 'pat_id').withColumnRenamed('dx_id_merge', 'dx_id').withColumnRenamed('instance_merge', 'instance')\
                                                .withColumnRenamed('full_diagnosis_name', 'diagnosis_name')

## Select only needed columns and then rename the column back to "noted_date" to keep following codes working
select_cols = ("pat_id", "dx_id", "instance", "diagnosis_name", "new_noted_date", "RESOLVED_DATE", "concept")
df_full_diagnosis_name = df_full_diagnosis_name.select(*select_cols).dropDuplicates()
diag_snomed_df = df_full_diagnosis_name.withColumnRenamed('new_noted_date', 'noted_date')


#convert snomed code 

diag_snomed_df = diag_snomed_df.where((lower(diag_snomed_df.concept).contains('snomed')))
###remove prefix
diag_snomed_df1 = diag_snomed_df.withColumn("SNOMED_ids", expr("substring(concept, 8, length(concept))"))

## rename RESOLVED_DATE to lowercase resolved_date
diag_snomed_df1 = diag_snomed_df1.withColumnRenamed("RESOLVED_DATE", "resolved_date")

diag_snomed_df1.createOrReplaceTempView('qw_diag_snomed')


######################################################################################################################################################################################################
## Function to find the presence of risk factors based on the SNOMED codes.
######################################################################################################################################################################################################
## Inputs:
## df: data frame for the target patient cohort 
##         Notice: need to assign the date used as reference with name "decided_index_date"
## list_of_risk_factors: keyword_list for finding risk factors parent snomed codes    
##               Check details of accepted keywords in this dictionary: https://adb-3942176072937821.1.azuredatabricks.net/?o=3942176072937821#notebook/1150766708331471
## only_instance_1k: a boolean, to control whether limit the diagnosis table to only have instance = 1000 records
## omop_table_version: the version of the OMOP table, check the data panel to see if there is a newer version
#######################################################################################################################################################################################################
## Output:
## df: data frame for the target patient cohort with each risk factors added as a binary feature
##     The updated dataframe with given risk facors
#######################################################################################################################################################################################################

def add_risk_factors_active_at_decided_index_date(df, list_of_risk_factors, only_instance_1k, omop_table_version):
  
  for risk_factor in list_of_risk_factors:
    ## Acquire current time, and convert to the right format
    now1 = datetime.now()
    start_time = now1.strftime("%H:%M:%S")
    
    print("The current risk factor in processing is {0}, start at {1}".format(risk_factor, start_time))
    codes = conditions_cc_registry[risk_factor]
    diagnosis_id_list, diagnosis_noDuplicate_id_list = [], []
    for code in codes:
      temp_df = get_all_descendant_snomed_codes(code, omop_table_version)
      # use toPandas to convert a pyspark dataframe's col into a list (faster than using flatmap)
      cur_snomed_list = list(temp_df.select('descendant_snomed_code').toPandas()['descendant_snomed_code'])
      diagnosis_id_list += cur_snomed_list
    
    ## Convert to set to make sure no duplicates    
    diagnosis_noDuplicate_id_list = list(set(diagnosis_id_list))
    
    if risk_factor in patches_cc_registry:
      list_to_ignore = patches_cc_registry[risk_factor]
      ## Following line only used to debug
      #print(list_to_ignore)
      ## Exclude snomed codes from the list_to_ignore
      diagnosis_noDuplicate_id_list = [ele for ele in diagnosis_noDuplicate_id_list if ele not in list_to_ignore]
      print("The special treatment to remove unwanted codes from {} is working!".format(risk_factor))
      ## Test only!!
#       print("Here is the list of snomed codes found: {}".format(diagnosis_noDuplicate_id_list))
    else:
      print("No special treatment to remove codes from {} now, please contact Jenn or the doctor you worked with to confirm.".format(risk_factor))
     
    ## Remove all strange strings start with OMOP
    def checkPrefix(x):
      prefix = "OMOP"
      if x.startswith(prefix):
        return False
      else: return True 
    new_diagnosis_codes = list(filter(checkPrefix, diagnosis_noDuplicate_id_list))
    
    diagnosis_ids = "','".join(new_diagnosis_codes)
    
    tmp =  spark.sql(
    """
    SELECT DISTINCT qw_diag_snomed.pat_id,qw_diag_snomed.instance, qw_diag_snomed.noted_date, qw_diag_snomed.resolved_date, IF(COUNT(diagnosis.dx_id) > 0, 1, 0) AS """ + risk_factor + """ 
    FROM ((qw_diag_snomed
      INNER JOIN rdp_phi.diagnosismapping ON qw_diag_snomed.dx_id = diagnosismapping.dx_id)
      INNER JOIN rdp_phi.diagnosis ON rdp_phi.diagnosismapping.dx_id = rdp_phi.diagnosis.dx_id)
    WHERE qw_diag_snomed.SNOMED_ids in ('""" + diagnosis_ids + """')
    GROUP BY qw_diag_snomed.pat_id, qw_diag_snomed.instance, qw_diag_snomed.noted_date, qw_diag_snomed.resolved_date
    """
    )
    
    if only_instance_1k:
      print("Filter to include only instance = 1000 records.")
      tmp = tmp.filter(tmp.instance == 1000)
    else:
      print("Include all instance numbers.")
    
    cond = (df.pat_id == tmp.pat_id) & (df.instance == tmp.instance) &\
           ( ( (df.decided_index_date >= tmp.noted_date) | (tmp.noted_date.isNull()) ) &\
           ( (df.decided_index_date <= tmp.resolved_date) | (tmp.resolved_date.isNull()) ) )

    df = df.join(tmp, cond, how='left').drop(tmp.pat_id).drop(tmp.instance).drop(tmp.noted_date).drop(tmp.resolved_date).fillna({risk_factor: 0})
    
    df = df.dropDuplicates()
    
    ## Get the number of patients found
    #num_pts = df.agg(F.sum(risk_factor)).collect()[0][0]
    
    ## Acquire current time, and convert to the right format
    now2 = datetime.now()
    td = now2 - now1
    td_mins = int(round(td.total_seconds() / 60))
    
    print("{0} finished, used approx. {1} minutes".format(risk_factor, td_mins))
  return df

