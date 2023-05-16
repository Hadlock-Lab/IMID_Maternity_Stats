# Author: Yeon Mi Hwang
# Date: 5/15/23

# Workflow of C_propensity_score_matching.py 
# 1. load necessary packages and functions 
# 2. (optional) Set seed value 
# 3. load cohorts (control, treatment)
# 4. format cohort dataframe for propensity score matching 
# 5. get propensity score  
# 6. matching 
# 7. evaluate matching 
# 8. save matched control cohort df 
# 9. individual IMID 


# 1. load necessary packages and functions  
import IMID_Maternity_Stats.utilities

# 2. (optional) set seed values

seed_value= 457
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 3. load cohorts 
IMID_maternity = spark.sql("SELECT * FROM rdp_phi_sandbox.qw_IMID_trainset_cond_med_20221231_IMID_maternity_fix")
IMID_maternity_2013 = IMID_maternity.filter(F.col('delivery_year')>2012)

imid_list = ['ibd', 'rheumatoid_arthritis', 'multiple_sclerosis', 'psoriatic_arthritis', 'psoriasis', 'systemic_sclerosis', 'spondyloarthritis', 'systemic_lupus', 'vasculitis','sarcoidosis', 'APS', 'sjogren_syndrome']
filter_string = ''
for i in range(0,len(imid_list)):
  if i != 0:
    string = ' OR {0} == 1'.format(imid_list[i])
  else:
    string = '{0} == 1'.format(imid_list[i])
  filter_string += string 


IMID_df = IMID_maternity_2013_full.filter(filter_string)
noIMID_df = IMID_maternity_2013_full.join(IMID_df, ['pat_id', 'instance'], 'left_anti')
IMID_df = IMID_df
 # 4. format cohort dataframe for propensity score matching 

 IMID_maternity_2013_full2 = IMID_maternity_2013_full.na.fill(value = 0, subset=["hydroxychloroquine_exposure0_status", "methotrexate_exposure0_status", "leflunomide_exposure0_status", "teriflunomide_exposure0_status", "Five_ASA_exposure0_status", "azathioprine_exposure0_status", "mercaptopurine_exposure0_status", "mitoxantrone_exposure0_status", "Calcineurin_inhibitor_exposure0_status", "TNF_inhibitor_exposure0_status", "fumarates_exposure0_status",
"interferons_exposure0_status","alkylating_exposure0_status","hydroxyurea_exposure0_status","dapsone_exposure0_status","cladribine_exposure0_status","IL1_inhibitor_exposure0_status","IL6_inhibitor_exposure0_status",
"IL12_23_inhibitor_exposure0_status","IL17_inhibitor_exposure0_status","IL23_inhibitor_exposure0_status","abatacept_exposure0_status","belimumab_exposure0_status","S1P_receptor_modulators_exposure0_status",
"JAKi_exposure0_status","Integrin_modulator_exposure0_status","PDE4i_targeted_synthetic_exposure0_status","cd20_exposure0_status","cd52_exposure0_status","budesonide_exposure0_status","steroids_exposure0_status","diabetes_type1and2", "chronic_kidney_disease", "obesity", "chronic_liver_disease",
"asthma", "HIV", "chronic_lung_disease", "depression", "hypercoagulability", "pneumonia", "urinary_tract_infection", "sexually_transmitted_disease", "periodontitis_disease", "cardiovascular_disease", "sickle_cell_disease", "sepsis"])


control_df = noIMID_df
experimental_df = IMID_df

SVI_score = ['RPL_THEMES', #overall tract summary ranking variable 
             'RPL_THEME1', #socioeconomic ranking variable 
             'RPL_THEME2', #household composition and disability 
             'RPL_THEME3', #minority status and language 
             'RPL_THEME4']  #housing type and transportation 

ruca_col = ['SecondaryRUCACode2010']


control_df = add_geo_features(control_df, 'svi2018_us', join_cols = ['pat_id', 'instance']).select(*(control_df.columns + SVI_score))
experimental_df= add_geo_features(experimental_df, 'svi2018_us', join_cols = ['pat_id', 'instance']).select(*(experimental_df.columns + SVI_score))
control_df = add_geo_features(control_df, 'ruca2010revised', join_cols = ['pat_id', 'instance']).select(*(control_df.columns + ruca_col))
experimental_df= add_geo_features(experimental_df, 'ruca2010revised', join_cols = ['pat_id', 'instance']).select(*(experimental_df.columns + ruca_col))
control_df = control_df.withColumn('ruca_categorization', categorize_ruca_udf(F.col('SecondaryRUCACode2010')))
experimental_df = experimental_df.withColumn('ruca_categorization', categorize_ruca_udf(F.col('SecondaryRUCACode2010')))

for svi in SVI_score:
  control_df = control_df.withColumn(svi, F.col(svi).cast(FloatType())).withColumn(svi, F.when(F.col(svi)<0, None).otherwise(F.col(svi)))
  experimental_df = experimental_df.withColumn(svi, F.col(svi).cast(FloatType())).withColumn(svi, F.when(F.col(svi)<0, None).otherwise(F.col(svi)))


select_columns = [ 'pregravid_bmi', 'age_at_start_dt', 'insurance', 'race','ethnic_group', 'ob_hx_infant_sex', 'Parity', 'Gravidity', 'Preterm_history' , 'illegal_drug_user', 'smoker', 'RPL_THEME1', 'RPL_THEME2','RPL_THEME3','RPL_THEME4', 'lmp','gestational_days', 'ruca_categorization', 'delivery_year',
## Cormorbidities
"diabetes_type1and2", "chronic_kidney_disease", "obesity", "chronic_liver_disease",
"asthma", "HIV", "chronic_lung_disease", "depression", "hypercoagulability", "pneumonia", "urinary_tract_infection", "sexually_transmitted_disease", "periodontitis_disease", "cardiovascular_disease", "sickle_cell_disease", "sepsis"]

matching_columns = intersection(control_df.columns, experimental_df.columns)

control_df2 = control_df.select(*matching_columns)
experimental_df2 = experimental_df.select(*matching_columns)

control_psm, covariates = format_dataframe_for_psm(control_df2, select_columns)


control_psm['IMID_status'] = 0
experimental_psm['IMID_status'] = 1
df_final = shuffle(control_psm.append(experimental_psm, ignore_index=True))



sa_select_columns = [ 'pregravid_bmi', 'age_at_start_dt', 'insurance', 'race','ethnic_group', 'ob_hx_infant_sex', 'Parity', 'Gravidity', 'Preterm_history' , 'illegal_drug_user', 'smoker', 'RPL_THEME1', 'RPL_THEME2','RPL_THEME3','RPL_THEME4', 'lmp','gestational_days', 'ruca_categorization', 'delivery_year']

matching_columns = intersection(control_df.columns, experimental_df.columns)

sa_control_df2 = control_df.select(*matching_columns)
sa_experimental_df2 = experimental_df.select(*matching_columns)

sa_control_psm, sa_covariates = format_dataframe_for_psm(sa_control_df2, sa_select_columns)
sa_experimental_psm, sa_covariates = format_dataframe_for_psm(sa_experimental_df2, sa_select_columns)

sa_control_psm['IMID_status'] = 0
sa_experimental_psm['IMID_status'] = 1
sa_df_final = shuffle(sa_control_psm.append(sa_experimental_psm, ignore_index=True))






# 5. get propensity score 

ps_IMID = get_propensity_score('IMID_status', covariates, df_final)
sa_ps_IMID = get_propensity_score('IMID_status', sa_covariates, sa_df_final)

# 6. match on propensity score 
df_final_matched = propensity_score_matching(ps_IMID, 'IMID_status')
sa_df_final_matched = propensity_score_matching(sa_ps_IMID, 'IMID_status')

# 7. evaluate matching 
from numpy import std, mean, sqrt

#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
  nx = len(x)
  ny = len(y)
  dof = nx + ny - 2
  return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)
def get_effect_size(original_df, matched_df, treatment, covariates):
  data = []
  cols = covariates 
  for cl in cols:
    data.append([cl,'before', cohen_d(original_df[original_df[treatment]==1][cl], original_df[original_df[treatment]==0][cl])])
    data.append([cl,'after', cohen_d(matched_df[matched_df[treatment]==1][cl], matched_df[matched_df[treatment]==0][cl])])
  res = pd.DataFrame(data, columns=['variable','matching','effect_size'])
  return res

 effect_size = get_effect_size(df_final, df_final_matched, 'IMID_status', covariates)
sa_effect_size = get_effect_size(sa_df_final, sa_df_final_matched, 'IMID_status', sa_covariates)
 # 8. save matched control df 
 matched_control = df_final_matched[df_final_matched['IMID_status']==0]
 df_temp = spark.createDataFrame(matched_control[['pat_id', 'instance', 'episode_id']])
df_control = noIMID_df
df_control.createOrReplaceTempView("control")
df_temp.createOrReplaceTempView("temp")
df_matched_final = spark.sql(
"""
SELECT c.*
FROM control AS c
INNER JOIN temp AS t 
ON c.pat_id = t.pat_id
  AND c.instance = t.instance
  AND c.episode_id = t.episode_id
  """)

table_name = 'rdp_phi_sandbox.yh_IMID_matched_control'
spark.sql("DROP TABLE IF EXISTS {0}".format(table_name))
df_matched_final.write.saveAsTable(table_name)

sa_matched_control = sa_df_final_matched[sa_df_final_matched['IMID_status']==0]
sa_df_temp = spark.createDataFrame(sa_matched_control[['pat_id', 'instance', 'episode_id']])
sa_df_control = noIMID_df
sa_df_control.createOrReplaceTempView("control")
sa_df_temp.createOrReplaceTempView("temp")
sa_df_matched_final = spark.sql(
"""
SELECT c.*
FROM control AS c
INNER JOIN temp AS t 
ON c.pat_id = t.pat_id
  AND c.instance = t.instance
  AND c.episode_id = t.episode_id
  """)

# 9. individual IMID 

IMID_df_dict = {}
for imid in imid_list:
  IMID_df_dict[imid] = experimental_df2.filter(F.col(imid)==1)

sa_IMID_df_dict = {}
for imid in imid_list:
  sa_IMID_df_dict[imid] = sa_experimental_df2.filter(F.col(imid)==1)

IMID_psm_dict = {}
for imid in imid_list:
  IMID_psm_dict[imid], covariates = format_dataframe_for_psm(IMID_df_dict[imid], select_columns)

sa_IMID_psm_dict = {}
for imid in imid_list:
  sa_IMID_psm_dict[imid], sa_covariates = format_dataframe_for_psm(sa_IMID_df_dict[imid], sa_select_columns)

control_imid_dict = {}
experimental_imid_dict = {}
final_imid_dict = {}
for imid in imid_list: 
  control_psm_copy = control_psm 
  control_psm_copy['{0}_status'.format(imid)] = 0
  control_imid_dict[imid] = control_psm_copy
  experimental_psm_copy = IMID_psm_dict[imid]
  experimental_psm_copy['{0}_status'.format(imid)] = 1
  experimental_imid_dict[imid] = experimental_psm_copy
  df_final_psm = shuffle(control_psm_copy.append(experimental_psm_copy, ignore_index=True))
  final_imid_dict[imid] = df_final_psm

sa_control_imid_dict = {}
sa_experimental_imid_dict = {}
sa_final_imid_dict = {}
for imid in imid_list: 
  sa_control_psm_copy = sa_control_psm 
  sa_control_psm_copy['{0}_status'.format(imid)] = 0
  sa_control_imid_dict[imid] = sa_control_psm_copy
  sa_experimental_psm_copy = sa_IMID_psm_dict[imid]
  sa_experimental_psm_copy['{0}_status'.format(imid)] = 1
  sa_experimental_imid_dict[imid] = sa_experimental_psm_copy
  sa_df_final_psm = shuffle(sa_control_psm_copy.append(sa_experimental_psm_copy, ignore_index=True))
  sa_final_imid_dict[imid] = sa_df_final_psm


ps_imid_dict = {}
matched_imid_dict = {}
for imid in imid_list:
  final_imid_dict[imid][covariates] = final_imid_dict[imid][covariates].fillna(value=0)
  ps_imid_dict[imid] = get_propensity_score('{0}_status'.format(imid), covariates, final_imid_dict[imid])
  matched_imid_dict[imid] = propensity_score_matching(ps_imid_dict[imid], '{0}_status'.format(imid))
  print (imid)

sa_ps_imid_dict = {}
sa_matched_imid_dict = {}
for imid in imid_list:
  sa_final_imid_dict[imid][sa_covariates] = sa_final_imid_dict[imid][sa_covariates].fillna(value=0)
  sa_ps_imid_dict[imid] = get_propensity_score('{0}_status'.format(imid), sa_covariates, sa_final_imid_dict[imid])
  sa_matched_imid_dict[imid] = propensity_score_matching(sa_ps_imid_dict[imid], '{0}_status'.format(imid))
  print (imid)



effect_size_df_dict = {}
effect_size_df_dict['variable'] = effect_size[effect_size.matching == 'after']['variable']
for key in effect_size_dict.keys():
  df = effect_size_dict[key]
  effect_size_df_dict[key] = round(df[df.matching == 'after']['effect_size'],2)


sa_effect_size_df_dict = {}
sa_effect_size_df_dict['variable'] = sa_effect_size[sa_effect_size.matching == 'after']['variable']
for key in sa_effect_size_dict.keys():
  df = sa_effect_size_dict[key]
  sa_effect_size_df_dict[key] = round(df[df.matching == 'after']['effect_size'],2)