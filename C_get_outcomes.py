# Author: Yeon Mi Hwang
# Date: 5/15/23


# Workflow of C_propensity_score_matching.py 
# 1. load necessary packages and functions 
# 2. load cohorts (control, treatment)
# 3. get outcomes - adjusted
# 4. get outcomes - unadjusted 
# 5. get outcomes - sensitivity analysis 



# 1. load necessary packages and functions  
import IMID_Maternity_Stats.utilities


# 2. load cohorts
IMID_maternity_2013 = IMID_maternity_2013.dropDuplicates(['pat_id', 'instance', 'lmp', 'gestational_days', 'ob_hx_infant_sex'])
imid_list = ['ibd', 'rheumatoid_arthritis', 'multiple_sclerosis', 'psoriatic_arthritis', 'psoriasis', 'systemic_sclerosis', 'spondyloarthritis', 'systemic_lupus', 'vasculitis','sarcoidosis', 'APS', 'sjogren_syndrome']
filter_string = ''
for i in range(0,len(imid_list)):
  if i != 0:
    string = ' OR {0} == 1'.format(imid_list[i])
  else:
    string = '{0} == 1'.format(imid_list[i])
  filter_string += string 


IMID_df = IMID_maternity_2013.filter(filter_string)
noIMID_df = IMID_maternity_2013.join(IMID_df, ['pat_id', 'instance'], 'left_anti')

noIMID_pd = noIMID_df.select(*['SGA','LBW','preterm_category', 'ibd',
 'rheumatoid_arthritis',
 'multiple_sclerosis',
 'psoriatic_arthritis',
 'psoriasis',
 'systemic_sclerosis',
 'spondyloarthritis',
 'systemic_lupus',
 'vasculitis',
 'sarcoidosis',
 'APS',
 'sjogren_syndrome', 'pregnancy_outcome', 'delivery_method']).toPandas()

matched_control_df = spark.sql("SELECT * FROM rdp_phi_sandbox.yh_IMID_matched_control").select(*['SGA','LBW','preterm_category', 'ibd',
 'rheumatoid_arthritis',
 'multiple_sclerosis',
 'psoriatic_arthritis',
 'psoriasis',
 'systemic_sclerosis',
 'spondyloarthritis',
 'systemic_lupus',
 'vasculitis',
 'sarcoidosis',
 'APS',
 'sjogren_syndrome', 'pregnancy_outcome', 'delivery_method']).toPandas()


imid_df_dict = {}
imid_matched_control_df_dict = {}
for imid in imid_list:
  matched_df = spark.sql("SELECT * FROM rdp_phi_sandbox.yh_IMID_{0}_matched_control".format(imid)).select(*['SGA','LBW','preterm_category', 'ibd',
 'rheumatoid_arthritis',
 'multiple_sclerosis',
 'psoriatic_arthritis',
 'psoriasis',
 'systemic_sclerosis',
 'spondyloarthritis',
 'systemic_lupus',
 'vasculitis',
 'sarcoidosis',
 'APS',
 'sjogren_syndrome', 'pregnancy_outcome', 'delivery_method']).toPandas()
  imid_df = IMID_df.filter(F.col(imid)==1).select(*['SGA','LBW','preterm_category', 'ibd',
 'rheumatoid_arthritis',
 'multiple_sclerosis',
 'psoriatic_arthritis',
 'psoriasis',
 'systemic_sclerosis',
 'spondyloarthritis',
 'systemic_lupus',
 'vasculitis',
 'sarcoidosis',
 'APS',
 'sjogren_syndrome', 'pregnancy_outcome', 'delivery_method']).toPandas()
  imid_df_dict[imid] = imid_df
  imid_matched_control_df_dict[imid] = matched_df


# 2. get outcomes - adjusted 
comparison = []
outcome = []
rr = []
lower = []
upper = []
pval = []
from scipy.stats.contingency import relative_risk
comparison += ['IMID', 'IMID', 'IMID', 'IMID']
imid_count = len(IMID_pd)
control_count = len(matched_control_df)
imid_preterm_count, imid_sga_count, imid_lbw_count, imid_csec_count = get_count(IMID_pd)
control_preterm_count, control_sga_count, control_lbw_count, control_csec_count = get_count(matched_control_df)
ptb_result = relative_risk(imid_preterm_count, imid_count, control_preterm_count, control_count)
ptb_p = fisher_exact([[imid_preterm_count, imid_count-imid_preterm_count], [control_preterm_count, control_count-control_preterm_count]], alternative='two-sided')[1]
sga_result = relative_risk(imid_sga_count, imid_count, control_sga_count, control_count)
sga_p = fisher_exact([[imid_sga_count, imid_count-imid_sga_count], [control_sga_count, control_count-control_sga_count]], alternative='two-sided')[1]
lbw_result = relative_risk(imid_lbw_count, imid_count, control_lbw_count, control_count)
lbw_p = fisher_exact([[imid_lbw_count, imid_count-imid_lbw_count], [control_lbw_count, control_count-control_lbw_count]], alternative='two-sided')[1]
csec_result = relative_risk(imid_csec_count, imid_count, control_csec_count, control_count)
csec_p = fisher_exact([[imid_csec_count, imid_count-imid_csec_count], [control_csec_count, control_count-control_csec_count]], alternative='two-sided')[1]
ptb_rr, ptb_rr_ci_l, ptb_rr_ci_u = ptb_result.relative_risk, ptb_result.confidence_interval(confidence_level=0.95)[0], ptb_result.confidence_interval(confidence_level=0.95)[1] 
sga_rr, sga_rr_ci_l, sga_rr_ci_u = sga_result.relative_risk, sga_result.confidence_interval(confidence_level=0.95)[0], sga_result.confidence_interval(confidence_level=0.95)[1] 
lbw_rr, lbw_rr_ci_l, lbw_rr_ci_u = lbw_result.relative_risk, lbw_result.confidence_interval(confidence_level=0.95)[0], lbw_result.confidence_interval(confidence_level=0.95)[1] 
csec_rr, csec_rr_ci_l, csec_rr_ci_u = csec_result.relative_risk, csec_result.confidence_interval(confidence_level=0.95)[0], csec_result.confidence_interval(confidence_level=0.95)[1] 
rr += [round(ptb_rr,2), round(lbw_rr,2), round(sga_rr,2), round(csec_rr,2)]
lower += [round(ptb_rr_ci_l,2), round(lbw_rr_ci_l,2), round(sga_rr_ci_l,2), round(csec_rr_ci_l,2)]
upper += [round(ptb_rr_ci_u,2), round(lbw_rr_ci_u,2), round(sga_rr_ci_u,2), round(csec_rr_ci_u,2)]
outcome += ['ptb', 'lbw', 'sga', 'csec']
pval += [get_pval_signal(ptb_p), get_pval_signal(lbw_p), get_pval_signal(sga_p), get_pval_signal(csec_p)]
for imid in imid_list:
  print (imid)
  comparison += [imid, imid, imid, imid]
  imid_count = len(imid_df_dict[imid])
  imid_preterm_count, imid_sga_count, imid_lbw_count, imid_csec_count = get_count(imid_df_dict[imid])
  control_count = len(imid_matched_control_df_dict[imid])
  control_preterm_count, control_sga_count, control_lbw_count, control_csec_count = get_count(imid_matched_control_df_dict[imid])
  ptb_result = relative_risk(imid_preterm_count, imid_count, control_preterm_count, control_count)
  ptb_p = fisher_exact([[imid_preterm_count, imid_count-imid_preterm_count], [control_preterm_count, control_count-control_preterm_count]], alternative='two-sided')[1]
  sga_result = relative_risk(imid_sga_count, imid_count, control_sga_count, control_count)
  sga_p = fisher_exact([[imid_sga_count, imid_count-imid_sga_count], [control_sga_count, control_count-control_sga_count]], alternative='two-sided')[1]
  lbw_result = relative_risk(imid_lbw_count, imid_count, control_lbw_count, control_count)
  lbw_p = fisher_exact([[imid_lbw_count, imid_count-imid_lbw_count], [control_lbw_count, control_count-control_lbw_count]], alternative='two-sided')[1]
  csec_result = relative_risk(imid_csec_count, imid_count, control_csec_count, control_count)
  csec_p = fisher_exact([[imid_csec_count, imid_count-imid_csec_count], [control_csec_count, control_count-control_csec_count]], alternative='two-sided')[1]
  ptb_rr, ptb_rr_ci_l, ptb_rr_ci_u = ptb_result.relative_risk, ptb_result.confidence_interval(confidence_level=0.95)[0], ptb_result.confidence_interval(confidence_level=0.95)[1] 
  sga_rr, sga_rr_ci_l, sga_rr_ci_u = sga_result.relative_risk, sga_result.confidence_interval(confidence_level=0.95)[0], sga_result.confidence_interval(confidence_level=0.95)[1] 
  lbw_rr, lbw_rr_ci_l, lbw_rr_ci_u = lbw_result.relative_risk, lbw_result.confidence_interval(confidence_level=0.95)[0], lbw_result.confidence_interval(confidence_level=0.95)[1] 
  csec_rr, csec_rr_ci_l, csec_rr_ci_u = csec_result.relative_risk, csec_result.confidence_interval(confidence_level=0.95)[0], csec_result.confidence_interval(confidence_level=0.95)[1] 
  rr += [round(ptb_rr,2), round(lbw_rr,2), round(sga_rr,2), round(csec_rr,2)]
  lower += [round(ptb_rr_ci_l,2), round(lbw_rr_ci_l,2), round(sga_rr_ci_l,2), round(csec_rr_ci_l,2)]
  upper += [round(ptb_rr_ci_u,2), round(lbw_rr_ci_u,2), round(sga_rr_ci_u,2), round(csec_rr_ci_u,2)]
  outcome += ['ptb', 'lbw', 'sga', 'csec']
  pval += [get_pval_signal(ptb_p), get_pval_signal(lbw_p), get_pval_signal(sga_p), get_pval_signal(csec_p)]


outcome_df_dict = {}
outcome_df_dict['comparison'] = comparison
outcome_df_dict['outcome'] = outcome
outcome_df_dict['rr'] = rr
outcome_df_dict['lower'] = lower
outcome_df_dict['upper'] = upper
outcome_df_dict['pval'] = pval
pd.DataFrame.from_dict(outcome_df_dict)


# 3. get outcomes - unadjusted
comparison = []
outcome = []
rr = []
lower = []
upper = []
pval = []
from scipy.stats.contingency import relative_risk
comparison += ['IMID', 'IMID', 'IMID', 'IMID']
imid_count = len(IMID_pd)
control_count = len(noIMID_pd)
imid_preterm_count, imid_sga_count, imid_lbw_count, imid_csec_count = get_count(IMID_pd)
control_preterm_count, control_sga_count, control_lbw_count, control_csec_count = get_count(noIMID_pd)
ptb_result = relative_risk(imid_preterm_count, imid_count, control_preterm_count, control_count)
ptb_p = fisher_exact([[imid_preterm_count, imid_count-imid_preterm_count], [control_preterm_count, control_count-control_preterm_count]], alternative='two-sided')[1]
sga_result = relative_risk(imid_sga_count, imid_count, control_sga_count, control_count)
sga_p = fisher_exact([[imid_sga_count, imid_count-imid_sga_count], [control_sga_count, control_count-control_sga_count]], alternative='two-sided')[1]
lbw_result = relative_risk(imid_lbw_count, imid_count, control_lbw_count, control_count)
lbw_p = fisher_exact([[imid_lbw_count, imid_count-imid_lbw_count], [control_lbw_count, control_count-control_lbw_count]], alternative='two-sided')[1]
csec_result = relative_risk(imid_csec_count, imid_count, control_csec_count, control_count)
csec_p = fisher_exact([[imid_csec_count, imid_count-imid_csec_count], [control_csec_count, control_count-control_csec_count]], alternative='two-sided')[1]
ptb_rr, ptb_rr_ci_l, ptb_rr_ci_u = ptb_result.relative_risk, ptb_result.confidence_interval(confidence_level=0.95)[0], ptb_result.confidence_interval(confidence_level=0.95)[1] 
sga_rr, sga_rr_ci_l, sga_rr_ci_u = sga_result.relative_risk, sga_result.confidence_interval(confidence_level=0.95)[0], sga_result.confidence_interval(confidence_level=0.95)[1] 
lbw_rr, lbw_rr_ci_l, lbw_rr_ci_u = lbw_result.relative_risk, lbw_result.confidence_interval(confidence_level=0.95)[0], lbw_result.confidence_interval(confidence_level=0.95)[1] 
csec_rr, csec_rr_ci_l, csec_rr_ci_u = csec_result.relative_risk, csec_result.confidence_interval(confidence_level=0.95)[0], csec_result.confidence_interval(confidence_level=0.95)[1] 
rr += [round(ptb_rr,2), round(lbw_rr,2), round(sga_rr,2), round(csec_rr,2)]
lower += [round(ptb_rr_ci_l,2), round(lbw_rr_ci_l,2), round(sga_rr_ci_l,2), round(csec_rr_ci_l,2)]
upper += [round(ptb_rr_ci_u,2), round(lbw_rr_ci_u,2), round(sga_rr_ci_u,2), round(csec_rr_ci_u,2)]
outcome += ['ptb', 'lbw', 'sga', 'csec']
pval += [get_pval_signal(ptb_p), get_pval_signal(lbw_p), get_pval_signal(sga_p), get_pval_signal(csec_p)]
for imid in imid_list:
  print (imid)
  comparison += [imid, imid, imid, imid]
  imid_count = len(imid_df_dict[imid])
  imid_preterm_count, imid_sga_count, imid_lbw_count, imid_csec_count = get_count(imid_df_dict[imid])
  ptb_result = relative_risk(imid_preterm_count, imid_count, control_preterm_count, control_count)
  ptb_p = fisher_exact([[imid_preterm_count, imid_count-imid_preterm_count], [control_preterm_count, control_count-control_preterm_count]], alternative='two-sided')[1]
  sga_result = relative_risk(imid_sga_count, imid_count, control_sga_count, control_count)
  sga_p = fisher_exact([[imid_sga_count, imid_count-imid_sga_count], [control_sga_count, control_count-control_sga_count]], alternative='two-sided')[1]
  lbw_result = relative_risk(imid_lbw_count, imid_count, control_lbw_count, control_count)
  lbw_p = fisher_exact([[imid_lbw_count, imid_count-imid_lbw_count], [control_lbw_count, control_count-control_lbw_count]], alternative='two-sided')[1]
  csec_result = relative_risk(imid_csec_count, imid_count, control_csec_count, control_count)
  csec_p = fisher_exact([[imid_csec_count, imid_count-imid_csec_count], [control_csec_count, control_count-control_csec_count]], alternative='two-sided')[1]
  ptb_rr, ptb_rr_ci_l, ptb_rr_ci_u = ptb_result.relative_risk, ptb_result.confidence_interval(confidence_level=0.95)[0], ptb_result.confidence_interval(confidence_level=0.95)[1] 
  sga_rr, sga_rr_ci_l, sga_rr_ci_u = sga_result.relative_risk, sga_result.confidence_interval(confidence_level=0.95)[0], sga_result.confidence_interval(confidence_level=0.95)[1] 
  lbw_rr, lbw_rr_ci_l, lbw_rr_ci_u = lbw_result.relative_risk, lbw_result.confidence_interval(confidence_level=0.95)[0], lbw_result.confidence_interval(confidence_level=0.95)[1] 
  csec_rr, csec_rr_ci_l, csec_rr_ci_u = csec_result.relative_risk, csec_result.confidence_interval(confidence_level=0.95)[0], csec_result.confidence_interval(confidence_level=0.95)[1] 
  rr += [round(ptb_rr,2), round(lbw_rr,2), round(sga_rr,2), round(csec_rr,2)]
  lower += [round(ptb_rr_ci_l,2), round(lbw_rr_ci_l,2), round(sga_rr_ci_l,2), round(csec_rr_ci_l,2)]
  upper += [round(ptb_rr_ci_u,2), round(lbw_rr_ci_u,2), round(sga_rr_ci_u,2), round(csec_rr_ci_u,2)]
  outcome += ['ptb', 'lbw', 'sga', 'csec']
  pval += [get_pval_signal(ptb_p), get_pval_signal(lbw_p), get_pval_signal(sga_p), get_pval_signal(csec_p)]
 
outcome_df_dict = {}
outcome_df_dict['comparison'] = comparison
outcome_df_dict['outcome'] = outcome
outcome_df_dict['rr'] = rr
outcome_df_dict['lower'] = lower
outcome_df_dict['upper'] = upper
outcome_df_dict['pval'] = pval
pd.DataFrame.from_dict(outcome_df_dict)  

# 3. get outcomes - sensitivity 

comparison = []
outcome = []
rr = []
lower = []
upper = []
pval = []
from scipy.stats.contingency import relative_risk
comparison += ['IMID', 'IMID', 'IMID', 'IMID']
imid_count = len(IMID_pd)
control_count = len(sa_matched_control_df)
imid_preterm_count, imid_sga_count, imid_lbw_count, imid_csec_count = get_count(IMID_pd)
control_preterm_count, control_sga_count, control_lbw_count, control_csec_count = get_count(sa_matched_control_df)
ptb_result = relative_risk(imid_preterm_count, imid_count, control_preterm_count, control_count)
ptb_p = fisher_exact([[imid_preterm_count, imid_count-imid_preterm_count], [control_preterm_count, control_count-control_preterm_count]], alternative='two-sided')[1]
sga_result = relative_risk(imid_sga_count, imid_count, control_sga_count, control_count)
sga_p = fisher_exact([[imid_sga_count, imid_count-imid_sga_count], [control_sga_count, control_count-control_sga_count]], alternative='two-sided')[1]
lbw_result = relative_risk(imid_lbw_count, imid_count, control_lbw_count, control_count)
lbw_p = fisher_exact([[imid_lbw_count, imid_count-imid_lbw_count], [control_lbw_count, control_count-control_lbw_count]], alternative='two-sided')[1]
csec_result = relative_risk(imid_csec_count, imid_count, control_csec_count, control_count)
csec_p = fisher_exact([[imid_csec_count, imid_count-imid_csec_count], [control_csec_count, control_count-control_csec_count]], alternative='two-sided')[1]
ptb_rr, ptb_rr_ci_l, ptb_rr_ci_u = ptb_result.relative_risk, ptb_result.confidence_interval(confidence_level=0.95)[0], ptb_result.confidence_interval(confidence_level=0.95)[1] 
sga_rr, sga_rr_ci_l, sga_rr_ci_u = sga_result.relative_risk, sga_result.confidence_interval(confidence_level=0.95)[0], sga_result.confidence_interval(confidence_level=0.95)[1] 
lbw_rr, lbw_rr_ci_l, lbw_rr_ci_u = lbw_result.relative_risk, lbw_result.confidence_interval(confidence_level=0.95)[0], lbw_result.confidence_interval(confidence_level=0.95)[1] 
csec_rr, csec_rr_ci_l, csec_rr_ci_u = csec_result.relative_risk, csec_result.confidence_interval(confidence_level=0.95)[0], csec_result.confidence_interval(confidence_level=0.95)[1] 
rr += [round(ptb_rr,2), round(lbw_rr,2), round(sga_rr,2), round(csec_rr,2)]
lower += [round(ptb_rr_ci_l,2), round(lbw_rr_ci_l,2), round(sga_rr_ci_l,2), round(csec_rr_ci_l,2)]
upper += [round(ptb_rr_ci_u,2), round(lbw_rr_ci_u,2), round(sga_rr_ci_u,2), round(csec_rr_ci_u,2)]
outcome += ['ptb', 'lbw', 'sga', 'csec']
pval += [get_pval_signal(ptb_p), get_pval_signal(lbw_p), get_pval_signal(sga_p), get_pval_signal(csec_p)]
for imid in imid_list:
  print (imid)
  comparison += [imid, imid, imid, imid]
  imid_count = len(imid_df_dict[imid])
  imid_preterm_count, imid_sga_count, imid_lbw_count, imid_csec_count = get_count(imid_df_dict[imid])
  control_count = len(sa_imid_matched_control_df_dict[imid])
  control_preterm_count, control_sga_count, control_lbw_count, control_csec_count = get_count(sa_imid_matched_control_df_dict[imid])
  ptb_result = relative_risk(imid_preterm_count, imid_count, control_preterm_count, control_count)
  ptb_p = fisher_exact([[imid_preterm_count, imid_count-imid_preterm_count], [control_preterm_count, control_count-control_preterm_count]], alternative='two-sided')[1]
  sga_result = relative_risk(imid_sga_count, imid_count, control_sga_count, control_count)
  sga_p = fisher_exact([[imid_sga_count, imid_count-imid_sga_count], [control_sga_count, control_count-control_sga_count]], alternative='two-sided')[1]
  lbw_result = relative_risk(imid_lbw_count, imid_count, control_lbw_count, control_count)
  lbw_p = fisher_exact([[imid_lbw_count, imid_count-imid_lbw_count], [control_lbw_count, control_count-control_lbw_count]], alternative='two-sided')[1]
  csec_result = relative_risk(imid_csec_count, imid_count, control_csec_count, control_count)
  csec_p = fisher_exact([[imid_csec_count, imid_count-imid_csec_count], [control_csec_count, control_count-control_csec_count]], alternative='two-sided')[1]  
  ptb_rr, ptb_rr_ci_l, ptb_rr_ci_u = ptb_result.relative_risk, ptb_result.confidence_interval(confidence_level=0.95)[0], ptb_result.confidence_interval(confidence_level=0.95)[1] 
  sga_rr, sga_rr_ci_l, sga_rr_ci_u = sga_result.relative_risk, sga_result.confidence_interval(confidence_level=0.95)[0], sga_result.confidence_interval(confidence_level=0.95)[1] 
  lbw_rr, lbw_rr_ci_l, lbw_rr_ci_u = lbw_result.relative_risk, lbw_result.confidence_interval(confidence_level=0.95)[0], lbw_result.confidence_interval(confidence_level=0.95)[1] 
  csec_rr, csec_rr_ci_l, csec_rr_ci_u = csec_result.relative_risk, csec_result.confidence_interval(confidence_level=0.95)[0], csec_result.confidence_interval(confidence_level=0.95)[1] 
  rr += [round(ptb_rr,2), round(lbw_rr,2), round(sga_rr,2), round(csec_rr,2)]
  lower += [round(ptb_rr_ci_l,2), round(lbw_rr_ci_l,2), round(sga_rr_ci_l,2), round(csec_rr_ci_l,2)]
  upper += [round(ptb_rr_ci_u,2), round(lbw_rr_ci_u,2), round(sga_rr_ci_u,2), round(csec_rr_ci_u,2)]
  outcome += ['ptb', 'lbw', 'sga', 'csec']
  pval += [get_pval_signal(ptb_p), get_pval_signal(lbw_p), get_pval_signal(sga_p), get_pval_signal(csec_p)]
 

outcome_df_dict = {}
outcome_df_dict['comparison'] = comparison
outcome_df_dict['outcome'] = outcome
outcome_df_dict['rr'] = rr
outcome_df_dict['lower'] = lower
outcome_df_dict['upper'] = upper
outcome_df_dict['pval'] = pval
pd.DataFrame.from_dict(outcome_df_dict)
