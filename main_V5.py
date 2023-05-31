# Project ML RWTH - Loan Granted?

'''
Classification problem. Automation of the loan eligibility process based on customer detail:
    
Loan_ID : Unique Loan ID
Gender : Male/ Female
Married : Applicant married (Y/N)
Dependents : Number of dependents 
Education : Applicant Education (Graduate/ Under Graduate)
Self_Employed : Self employed (Y/N)
ApplicantIncome : Applicant income
CoapplicantIncome : Coapplicant income
LoanAmount : Loan amount in thousands of dollars
Loan_Amount_Term : Term of loan in months
Credit_History : credit history meets guidelines yes or no
Property_Area : Urban/ Semi Urban/ Rural
Loan_Status : Loan approved (Y/N) this is the target variable

'''
#%%

import functions
import subprocess
subprocess.call(['python', 'functions.py'])

#%%
# Load dataframe and set data types. 
import pandas as pd

df_original = pd.read_csv(r'C:\Users\marti\OneDrive\Desktop\ML_RWTH\loan\data.csv')
df = df_original.copy()
columns = df.columns.to_list()

#Data type evaluation
(df.dtypes).value_counts().plot(kind="bar")  # Categorical, boolean and numeric variables but there are some variable-types not loaded correctly. 
df = df.astype({
    'Gender': 'category',
    'Married': 'category',
    'Dependents': 'category',
    'Education': 'category',
    'Self_Employed': 'category',
    'Credit_History': 'category',
    'Property_Area': 'category',
    'Loan_Status': 'category',
    'Loan_Amount_Term': 'category',
    'ApplicantIncome': 'float64'
    })
df.info()

#%%
# Distribution of numerical variables

functions.UVA_numeric(df,['ApplicantIncome'])
functions.UVA_numeric(df,['CoapplicantIncome'])
functions.UVA_numeric(df,['LoanAmount'])

#%%
# Normality check of numerical variables

functions.normality_check(df,['ApplicantIncome'])
functions.normality_check(df,['CoapplicantIncome'])
functions.normality_check(df,['LoanAmount'])

#%%
# Category variable analysis

functions.UVA_category(df,['Gender','Married','Dependents'])
functions.UVA_category(df,['Education','Self_Employed'])
functions.UVA_category(df,['Loan_Amount_Term','Credit_History','Property_Area'])
functions.UVA_category(df,['Property_Area','Loan_Status'])

#Categorical variables (The categories in the categorical variables seems to be ok, including the 3+.). 
    #81% are males, 78% are graduated, 86% non self-employed, 85% loan time of 30 years, 84% with credit history
    #69% loan status YES and 31% NO. PENDING: Evaluate if a balance is needed. 
    
# Categorical analysis
functions.BVA_categorical_plot(df, 'Gender', 'Loan_Status')
functions.BVA_categorical_plot(df, 'Married', 'Loan_Status')
functions.BVA_categorical_plot(df, 'Property_Area', 'Loan_Status')
functions.BVA_categorical_plot(df, 'Dependents', 'Loan_Status')
functions.BVA_categorical_plot(df, 'Credit_History', 'Loan_Status')


#%%
# Outliers study

functions.UVA_outlier(df,['ApplicantIncome'])
functions.UVA_outlier(df,['ApplicantIncome'],include_outlier=False)
functions.UVA_outlier(df,['CoapplicantIncome'])
functions.UVA_outlier(df,['CoapplicantIncome'],include_outlier=False)
functions.UVA_outlier(df,['LoanAmount'])
functions.UVA_outlier(df,['LoanAmount'],include_outlier=False)

# NAs treatment
df.isna().sum()

# Distribution outliers
distribution, df_outliers = functions.Outliers_Distribution (df)    
# df = df.drop(df_outliers.index) -- to eliminate outliers
# Outliers have a similar distribution regarding to Loan Status (63% Y - 37% N) 


#%%
# Training ML models
results = functions.ML_Models(df)

#%%
# Creating new dataframe with manual features

dff = functions.create_manual_features (df)
results_dff = functions.ML_Models(dff)

#%%
# Creating new dataframe with automatic new features

dff2 = functions.create_new_features(df, degree=2)
results_dff2 = functions.ML_Models(dff2)

#%%
# Training a NN network
test_accuracy, f1, cm = functions.NN_Model (df)





