# IMIDs_Maternity_Stats

This contains the code used to conduct the retrospective cohort study for Hwang et al. 2023 manuscript on Immune mediated inflammatory disease and risk of adverse pregnancy outcomes.

All clinical logic is shared. Results were aggregated and reported within the paper to the extent possible while maintaining privacy from personal health information (PHI) as required by law. All data is archived within PHS systems in a HIPAA-secure audited compute environment to facilitate verification of study conclusions. Due to the sensitive nature of the data we are unable to share the data used in these analyses, although, all scripts used to conduct the analyses of the paper are shared herein.

* For codes of ETL tools, please refer to https://github.com/Hadlock-Lab/CEDA_tools_ver20221220


## Installation
We used Python version 3.8.10. 

## Workflow 
Our workflow is described using alphabets. 

- [utilities](https://github.com/Hadlock-Lab/IMIDs_Maternity_Stats/tree/main/utilities.py) contains functions written by Hadlock Lab and required to be loaded for analysis   

- [A_cohort_prepartion](https://github.com/Hadlock-Lab/IMIDs_Maternity_Stats/blob/main/A_cohort_prepartion.py) prepares cohort for analysis. Includes cohort selection process and feature engineering. 

- [B_propensity_score_matching](https://github.com/Hadlock-Lab/IMIDs_Maternity_Stats/blob/main/B_propensity_score_matching.py) runs propensity score matching and sensitivity analysis. 

- [C_get_outcomes](https://github.com/Hadlock-Lab/IMIDs_Maternity_Stats/blob/main/C_get_outcomes.py)calculate risk ratio and p-value. 
