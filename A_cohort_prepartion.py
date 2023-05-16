# Author: Qi Wei, Yeon Mi Hwang  
# Date: 5/15/23

# Workflow of A_cohort preparation.py 
# 1. load necessary packages and functions 
# 2. generate cohort 
# 3. add medication information 


# 1. load necessary packages and functions  
import IMID_Maternity_Stats.utilities


from pyspark.sql.functions import lower, col, lit, when, unix_timestamp, months_between, expr, countDistinct, count
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, FloatType


# 2. generate cohort 

pts_cohort = spark.sql("SELECT * FROM rdp_phi_sandbox.yh_maternity_2022")


pts_cohort = pts_cohort.withColumn("decided_index_date", col("lmp"))


list_of_risk_factors = [
                        ## Comorbidities
                        'hypertension_pregnancy_related', 'diabetes_type1and2', 'atrial_fibrillation', 'coronary_artery_disease', 'heart_failure', 'chronic_kidney_disease', 'copd', 'obesity', 'chronic_liver_disease', 'malignant_neoplastic_disease',                     
                        ## Added based on CDC website
                        'asthma', 'HIV', 'history_transplant', 'stroke', 'opioid_dependence',
                        ## Additional conditions
                        'abdominal_pain', 'acute_pulmonary_edema', 'acute_renal_failure', 'alcoholism', 'acute_respiratory_failure', 'arteriosclerotic_vascular_disease', 'benign_prostatic_hyperplasia', 'blood_loss', 'cardiac_arrhythmia', 'cardiorespiratory_failure', 'cdifficile_infection', 'cerebrovascular_accident', 'chronic_lung_disease', 'chronic_renal_failure', 'cirrhosis_liver_cooccurrent', 'cogan_syndrome', 'cold_agglutinin_disease', 'congestive_heart_failure', 'deep_vein_thrombosis', 'dementia', 'depression', 'dermatomyositis', 'diverticulitis', 'encephalopathy', 'endometriosis', 'gerd', 'gestational_trophoblastic_disease', 'graft_versus_host_disease', 'HBV', 'HCV', 'HDV',  'HEV', 'hidradenitis_suppurativa', 'history_bariatric', 'hypercoagulability', 'hyperlipidemia', 'hypothyroidism', 'ibs', 'insomnia', 'malabsorption', 'multicentric_reticulohistiocytosis', 'myastenia',  'myelodysplastic', 'NAFLD', 'neutropenia', 'osteoporosis', 'pancreatitis', 'pneumonia', 'polymyositis', 'pregnancy_tubal_ectopic', 'protein_calorie_malnutrition', 'pulmonary_embolism', 'schizophrenia',  'seizure', 'spinal_enthesopathy', 'tachycardia', 'urinary_tract_infection', 'wernicke_korsakoff',
                        ## IMIDs
                        ## Removed list: 'uveitis'
                        'ibd', 'rheumatoid_arthritis', 'multiple_sclerosis', 'psoriatic_arthritis', 'psoriasis', 'systemic_sclerosis', 'spondyloarthritis', 'systemic_lupus', 'vasculitis','sarcoidosis', 'APS', 'sjogren_syndrome'  
                       ]

## Add additional risk factors
## atopic dermititis, eczema, Vitiligo, ivt

only_instance_1k = False
omop_table_version = "2022_11_07"

condition_df = add_risk_factors_active_at_decided_index_date(pts_cohort, list_of_risk_factors, only_instance_1k, omop_table_version)

## Print the total number of distinct patient_id records in the df
print("Total number of records in df:", condition_df.select("pat_id").distinct().count())

## Print the total number of rows in the df
print("Total number of records in df:", condition_df.count())


spark.sql("""DROP TABLE IF EXISTS rdp_phi_sandbox.qw_IMID_trainset_cond_cohort_{}_fix""".format(file_date))
table_name = "rdp_phi_sandbox.qw_IMID_trainset_cond_cohort_{}_fix".format(file_date)
condition_df.write.mode("overwrite").format("delta").saveAsTable(table_name)


## Read the table
imids_cohort = spark.sql("SELECT * FROM rdp_phi_sandbox.qw_IMID_trainset_cond_cohort_{}_fix".format(file_date))

## Print the total number of distinct patient_id records in the df
print("Total number of records in df:", imids_cohort.select("pat_id").distinct().count())

## Print the total number of rows in the df
print("Total number of records in df:", imids_cohort.count())


# 3. add medication information 
med_list = ['hydroxychloroquine', 'methotrexate', 'leflunomide', 'teriflunomide', 'Five_ASA', 'azathioprine', 'mercaptopurine', 'mitoxantrone', 'mycophenolate', 'Calcineurin_inhibitor', 'TNF_inhibitor', 'fumarates', 'interferons', 'alkylating', 'hydroxyurea', 'dapsone', 'cladribine', 'IL1_inhibitor', 'IL6_inhibitor', 'IL12_23_inhibitor', 'IL17_inhibitor','IL23_inhibitor', 'abatacept', 'belimumab', 'S1P_receptor_modulators', 'JAKi', 'Integrin_modulator', 'PDE4i_targeted_synthetic', 'cd20', 'cd52', 'budesonide', 'steroids']
med_filter_string = ''
for i in range(0,len(med_list)):
  if i != 0:
    string = ' OR {0} == 1'.format(med_list[i])
  else:
    string = '{0} == 1'.format(med_list[i])
  med_filter_string += string 

 possible_routes = ['Oral','Intramuscular', 'Intravenous', 'Subcutaneous Infusion', 'Subcutaneous', 'Intravenous (Continuous Infusion)', 'Rectal']


 full_meds = get_medication_orders(cohort_df=IMID_maternity_2013.select(*['pat_id', 'instance','gestational_days', 'lmp', 'ob_delivery_delivery_date']), filter_string='end_date > lmp AND start_date < ob_delivery_delivery_date' ,add_cc_columns=med_list).filter(med_filter_string).filter(F.col('order_class')!='Historical Med').drop(*['order_description', 'order_class', 'order_set', 'order_status', 'order_priority', 'requested_instant_utc', 'authorizing_prov_id', 'department_id', 'controlled', 'recorded_datetime', 'due_datetime', 'scheduled_on_datetime', 'scheduled_for_datetime']).distinct()
 meds = get_medication_orders(cohort_df=IMID_df.select(*['pat_id', 'instance','gestational_days', 'lmp', 'ob_delivery_delivery_date']), filter_string='end_date > lmp AND start_date < ob_delivery_delivery_date' ,add_cc_columns=med_list).filter(med_filter_string).filter(F.col('order_class')!='Historical Med').drop(*['order_description', 'order_class', 'order_set', 'order_status', 'order_priority', 'requested_instant_utc', 'authorizing_prov_id', 'department_id', 'controlled', 'recorded_datetime', 'due_datetime', 'scheduled_on_datetime', 'scheduled_for_datetime']).distinct().filter(F.col('route').isin(possible_routes))
 meds = get_medinfo_cohort(meds)
 full_meds = get_medinfo_cohort(full_meds)

 from functools import reduce
for med in med_list:
  meds = meds.withColumn( med , F.col(med).cast("int"))
  full_meds = full_meds.withColumn( med , F.col(med).cast("int"))

full_meds_pd = full_meds.toPandas()
meds_pd = meds.toPandas()
full_meds_pd['med_name'] = full_meds_pd[med_list].idxmax(axis = "columns")
meds_pd['med_name'] = meds_pd[med_list].idxmax(axis = "columns")
full_meds_df=spark.createDataFrame(full_meds_pd) 
meds_df=spark.createDataFrame(meds_pd) 


partition_columns = ['pat_id', 'instance', 'lmp', 'gestational_days', 'med_name']
agg_cols = {'exposure_total_duration':'sum',
           'exposure1_duration':'sum',
           'exposure2_duration':'sum',
           'exposure3_duration':'sum',
           'exposure_total_status' :'max',
           'exposure0_status':'max',
           'exposure1_status':'max',
           'exposure2_status':'max',
           'exposure3_status':'max'}
full_med_agg = aggregate_data(full_meds_df,
                          partition_columns = partition_columns, 
                          aggregation_columns = agg_cols)


partition_columns = ['pat_id', 'instance', 'lmp', 'gestational_days', 'med_name']
agg_cols = {'exposure_total_duration':'sum',
           'exposure1_duration':'sum',
           'exposure2_duration':'sum',
           'exposure3_duration':'sum',
           'exposure_total_status' :'max',
           'exposure0_status':'max',
           'exposure1_status':'max',
           'exposure2_status':'max',
           'exposure3_status':'max'}
med_agg = aggregate_data(meds_df,
                          partition_columns = partition_columns, 
                          aggregation_columns = agg_cols)

for med in med_list:
  full_med_agg = full_med_agg.withColumn('{0}_exposure_total_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure_total_status_max')==1), 1).otherwise(0))
  full_med_agg = full_med_agg.withColumn('{0}_exposure0_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure0_status_max')==1), 1).otherwise(0))
  full_med_agg = full_med_agg.withColumn('{0}_exposure1_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure1_status_max')==1), 1).otherwise(0))
  full_med_agg = full_med_agg.withColumn('{0}_exposure2_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure2_status_max')==1), 1).otherwise(0))
  full_med_agg = full_med_agg.withColumn('{0}_exposure3_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure3_status_max')==1), 1).otherwise(0))

 for med in med_list:
  med_agg = med_agg.withColumn('{0}_exposure_total_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure_total_status_max')==1), 1).otherwise(0))
  med_agg = med_agg.withColumn('{0}_exposure0_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure0_status_max')==1), 1).otherwise(0))
  med_agg = med_agg.withColumn('{0}_exposure1_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure1_status_max')==1), 1).otherwise(0))
  med_agg = med_agg.withColumn('{0}_exposure2_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure2_status_max')==1), 1).otherwise(0))
  med_agg = med_agg.withColumn('{0}_exposure3_status'.format(med), F.when((F.col('med_name')== med)&(F.col('exposure3_status_max')==1), 1).otherwise(0))


partition_columns = ['pat_id', 'instance', 'lmp', 'gestational_days']
agg_cols = {}
for med in med_list:
  agg_cols['{0}_exposure_total_status'.format(med)] = 'max'
  agg_cols['{0}_exposure0_status'.format(med)] = 'max'
  agg_cols['{0}_exposure1_status'.format(med)] = 'max'
  agg_cols['{0}_exposure2_status'.format(med)] = 'max'
  agg_cols['{0}_exposure3_status'.format(med)] = 'max'
full_med_agg2 = aggregate_data(full_med_agg,
                          partition_columns = partition_columns, 
                          aggregation_columns = agg_cols)
for med in med_list:
  full_med_agg2 = full_med_agg2.withColumnRenamed('{0}_exposure_total_status_max'.format(med), '{0}_exposure_total_status'.format(med))
  full_med_agg2 = full_med_agg2.withColumnRenamed('{0}_exposure0_status_max'.format(med), '{0}_exposure0_status'.format(med))
  full_med_agg2 = full_med_agg2.withColumnRenamed('{0}_exposure1_status_max'.format(med), '{0}_exposure1_status'.format(med))
  full_med_agg2 = full_med_agg2.withColumnRenamed('{0}_exposure2_status_max'.format(med), '{0}_exposure2_status'.format(med))
  full_med_agg2 = full_med_agg2.withColumnRenamed('{0}_exposure3_status_max'.format(med), '{0}_exposure3_status'.format(med))


partition_columns = ['pat_id', 'instance', 'lmp', 'gestational_days']
agg_cols = {}
for med in med_list:
  agg_cols['{0}_exposure_total_status'.format(med)] = 'max'
  agg_cols['{0}_exposure0_status'.format(med)] = 'max'
  agg_cols['{0}_exposure1_status'.format(med)] = 'max'
  agg_cols['{0}_exposure2_status'.format(med)] = 'max'
  agg_cols['{0}_exposure3_status'.format(med)] = 'max'
med_agg2 = aggregate_data(med_agg,
                          partition_columns = partition_columns, 
                          aggregation_columns = agg_cols)

for med in med_list:
  med_agg2 = med_agg2.withColumnRenamed('{0}_exposure_total_status_max'.format(med), '{0}_exposure_total_status'.format(med))
  med_agg2 = med_agg2.withColumnRenamed('{0}_exposure0_status_max'.format(med), '{0}_exposure0_status'.format(med))
  med_agg2 = med_agg2.withColumnRenamed('{0}_exposure1_status_max'.format(med), '{0}_exposure1_status'.format(med))
  med_agg2 = med_agg2.withColumnRenamed('{0}_exposure2_status_max'.format(med), '{0}_exposure2_status'.format(med))
  med_agg2 = med_agg2.withColumnRenamed('{0}_exposure3_status_max'.format(med), '{0}_exposure3_status'.format(med))


 

