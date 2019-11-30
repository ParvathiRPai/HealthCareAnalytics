# Databricks notebook source
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

df=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/Insurance.csv')

# COMMAND ----------

#dataExploration
df.shape

# COMMAND ----------

df.columns

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.head()

# COMMAND ----------

df.describe()

# COMMAND ----------

states=df["State"].value_counts()
plans=df["Plan ID - Standard Component"].value_counts()
states

# COMMAND ----------

no_of_states=len(states)
no_of_plans=len(plans)
no_of_states, no_of_plans

# COMMAND ----------

# First Finding: Not all states make use of the federal network healthcare.gov. This may be due to the fact that several states have their own health insurance marketplace (like NY, for example).
# Second Finding: Some states offer significantly more plans than others. This may be due to the different sizes of the states.

# COMMAND ----------

unique_insurance=df['Issuer Name'].unique()
unique_insurance

# COMMAND ----------

unique_insurance_count=len(unique_insurance)
unique_insurance_count

# COMMAND ----------


Statelist = df['State'].unique()
Statelist = np.sort(Statelist)

# COMMAND ----------

plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")
sns.boxplot(x="State", y="Couple+1 child, Age 30",data=df, order=Statelist)
display()

# COMMAND ----------

df_VA=df[df['State'].isin(['VA'])].copy()
df_VA.shape

# COMMAND ----------

df_VA.describe()

# COMMAND ----------

VA_unique_Insurance=df_VA['Issuer Name'].unique

# COMMAND ----------

VA_unique_Insurance_count=len('VA_unique_Insurance')
VA_unique_Insurance_count

# COMMAND ----------

df.head()

# COMMAND ----------

median=pd.DataFrame(df.groupby('Plan ID - Standard Component',sort=False)['Premium Adult Individual Age 21','Premium Adult Individual Age 27','Premium Adult Individual Age 30','Premium Adult Individual Age 40','Premium Adult Individual Age 50','Premium Adult Individual Age 60'].median())
median=median.rename(columns={'Premium Adult Individual Age 21':'median_Premium Adult Individual Age 21','Premium Adult Individual Age 27':'median_Premium Adult Individual Age 27','Premium Adult Individual Age 30':'median_Premium Adult Individual Age 30','Premium Adult Individual Age 40':'median_Premium Adult Individual Age 40','Premium Adult Individual Age 50':'median_Premium Adult Individual Age 50','Premium Adult Individual Age 60':'median_Premium Adult Individual Age 60'})
median.head(3)

# COMMAND ----------



# COMMAND ----------

