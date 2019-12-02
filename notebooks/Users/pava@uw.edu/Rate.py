# Databricks notebook source
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# Data Exploration
df=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/Rate.csv')

# COMMAND ----------

df.shape

# COMMAND ----------

df.columns

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.head()

# COMMAND ----------

# Summarization of the 12 million rows.
df.describe()

# COMMAND ----------

# Based on the first look at the data, the intresting information is Business Year, State Code, Age, PlanId, Individual rate , Individual Tobacco rate and family/Couple rates
states = df["StateCode"].value_counts()
plans = df["PlanId"].value_counts()
states

# COMMAND ----------

no_of_states = len(states)
no_of_plans = len(plans)
no_of_states, no_of_plans
# we have data from 39 states covering 16808 insurance plans

# COMMAND ----------

#Data_cleaning:
df['Age'].unique()

# COMMAND ----------

df['IndividualRate'].plot.hist()
display()

# COMMAND ----------

#IndividualRate is varying. Lets try to find why?
df.sort_values(by='IndividualRate')

# COMMAND ----------

# removing NaN values
na_values = ['NaN', 'N/A', '0', '0.01', '9999', '9999.99', '999999']
df = pd.read_csv("/dbfs/autumn_2019/pava/ProjectData/Rate.csv", na_values=na_values, usecols=['BusinessYear', 'StateCode', 'PlanId', 'RatingAreaId','Tobacco', 'Age', 'IndividualRate','IndividualTobaccoRate','Couple', 'PrimarySubscriberAndOneDependent', 'PrimarySubscriberAndTwoDependents', 'PrimarySubscriberAndThreeOrMoreDependents', 'CoupleAndOneDependent', 'CoupleAndTwoDependents', 'CoupleAndThreeOrMoreDependents'])

# COMMAND ----------

df = df.drop_duplicates().reset_index(drop=True)

# COMMAND ----------

df.shape

# COMMAND ----------

# the raw data set contained 250,000 redundant rows in the dataframe and 24 columns are reduced to 15 columns

# COMMAND ----------

df["BusinessYear"].value_counts()
# the data is distrubuted for three business years and evenly distributed

# COMMAND ----------

# Data trends over the years
sns.boxplot(x="BusinessYear", y="IndividualRate", data=df)
display()

# COMMAND ----------

# something is wrong in 2014 lets break down the data in 2014
Statelist = df['StateCode'].unique()
Statelist = np.sort(Statelist)

# COMMAND ----------

plt.figure(figsize=(15, 6))
sns.boxplot(x="StateCode", y="IndividualRate", data=df, order=stateList)
display()

# COMMAND ----------

# Something is wrong with Virginia
# Lets split the dataFrane into three separate data frames as per years
df2014 = df[df['BusinessYear'].isin([2014])].copy()
df2015 = df[df['BusinessYear'].isin([2015])].copy()
df2016 = df[df['BusinessYear'].isin([2016])].copy()

# COMMAND ----------

df2014.describe()

# COMMAND ----------

# lets get 2014 VA copy of data
df2014_va = df2014[df2014['StateCode'].isin(['VA'])].copy()

# COMMAND ----------

df2014_va.shape

# COMMAND ----------

df2014_va.describe()

# COMMAND ----------

# May be high rates in VA is due to many old people. Lets see the the prices based on age 
list = df2014_va['Age'].isin(['Family Option'])
df2014_va_wofamily = df2014_va[~list]
age_labels = df2014_va_wofamily['Age'].unique()
age_labels = ['65+' if x=='65 and over' else x for x in age_labels]  #replace label '65 and over' with '65+' for plot labels
len(age_labels), age_labels

# COMMAND ----------

fig = plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")

fig.suptitle('Rates offered through Healthcare.gov in 2014 in Virginia', fontsize=14)

ax = sns.boxplot(x="Age", y="IndividualRate", data=df2014_va_wofamily, linewidth=1.0, fliersize=2.0)
ax.set_ylabel("Monthly rate in USD")

xticks = np.arange(46)
ax.xaxis.set_ticks(xticks)
ax.set_xticklabels(age_labels)

plt.savefig('Virginia_rates_by_age.png', bbox_inches='tight', dpi=150)
display()

# COMMAND ----------

# The large spread in the monthy rate of plans is some features are not included in VA health plan (http://www.webmd.com/health-insurance/20131011/why-some-virginia-health-plans-cost-so-much)

# COMMAND ----------

