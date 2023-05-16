# Author: Yeon Mi Hwang
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