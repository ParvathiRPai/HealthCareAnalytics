# Databricks notebook source
# MAGIC %md
# MAGIC #Health Care Analytics
# MAGIC ###Summary:
# MAGIC * Health care industry is huge and there is lot of data coming out of health care.
# MAGIC * Overall spending is $3.8 Trillion per year in US health care. 
# MAGIC * The estimated waste is $765 billion dollors
# MAGIC * Not only cost is issue, but health care is poor. 
# MAGIC * In this project we are focusing on analysis of US health care and proving solutions that provide better health care at lower cost

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 1: Average cost of insurance in 50 States
# MAGIC * For this analysis I used Inpatient prospective payment system from dava.gov site
# MAGIC * The data includes hospital specific charges for 3000 US hospitals which covers 50 states.
# MAGIC * Average covered charge: The average charge billed to the medicare by the provider.
# MAGIC * Average total payments: The total payments made to the provider (including payments from Medicare as well as the co-payments and deductibles paid by the beneficiary)
# MAGIC * Average Medicare Payments: The average payment just from Medicare.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_cost=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/IPPS_Data_Clean.csv')
df_cost.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Billed charges vary hopital to hospital

# COMMAND ----------

nationalmedian = pd.DataFrame(df_cost.groupby('drg_id',sort=False)['average_covered_charges','average_total_payments','average_medicare_payments'].median()).reset_index()
nationalmedian = nationalmedian.rename(columns={'average_covered_charges':'median_covered_charges', 'average_total_payments':'median_total_payments','average_medicare_payments':'median_medicare_payments' })
nationalmedian.head()

# COMMAND ----------

costoftreatmentperstate=df_cost.groupby(['drg_id','provider_state']).size()
costoftreatmentperstate =costoftreatmentperstate.unstack('provider_state').fillna(0)
costoftreatmentperstate.sum().plot(ylim=0,kind='bar',figsize=(25,8),fontsize=16);
display()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 2:
# MAGIC * Hospital ownership (Government, private, non-profit) and average rating
# MAGIC * The average rating is based on hospital overall rating, saftey of care national rating, patient experience national comparison.
# MAGIC * Hospital rating in 51 states 

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy as sp
df_hospitalrating=pd.read_csv("/dbfs/autumn_2019/pava/ProjectData/HospInfo.csv")
df_ratings=df_hospitalrating.drop(['Address','ZIP Code','Phone Number','Emergency Services',
'Patient experience national comparison footnote','Efficient use of medical imaging national comparison footnote',
'Hospital overall rating footnote','Mortality national comparison footnote','Mortality national comparison footnote',
'Safety of care national comparison footnote','Readmission national comparison footnote',
'Patient experience national comparison footnote','Effectiveness of care national comparison footnote',
'Timeliness of care national comparison footnote','Timeliness of care national comparison footnote',
'Efficient use of medical imaging national comparison footnote','Meets criteria for meaningful use of EHRs',
'Provider ID'], axis=1)
df1=df_ratings.replace('Below the National average',1)
df2=df1.replace('Same as the National average',2)
df3=df2.replace('Above the National average',3)
df3['Hospital overall rating'] = df3['Hospital overall rating'].convert_objects(convert_numeric=True)
sns.barplot(x='Hospital overall rating', y='Hospital Ownership',ci=0, data=df3, palette="deep")
plt.title('Hospital Ownership & Average Rating')
plt.xlabel('Rating')
plt.ylabel('Ownership Category')
plt.tight_layout()
display()

# COMMAND ----------

df1=df_ratings.replace('Below the National average',1) 
df2=df1.replace('Same as the National average',2)
df3=df2.replace('Above the National average',3)


# Convert types to numeric:
df3['Mortality national comparison'] = df3['Mortality national comparison'].convert_objects(convert_numeric=True)
df3['Safety of care national comparison'] = df3['Safety of care national comparison'].convert_objects(convert_numeric=True)
df3['Readmission national comparison'] = df3['Readmission national comparison'].convert_objects(convert_numeric=True)
df3['Patient experience national comparison'] = df3['Patient experience national comparison'].convert_objects(convert_numeric=True)
df3['Mortality national comparison'] = df3['Mortality national comparison'].convert_objects(convert_numeric=True)
df3['Hospital overall rating'] = df3['Hospital overall rating'].convert_objects(convert_numeric=True)

df5=df3.pivot_table(index=['State'], values = ['Hospital overall rating','Safety of care national comparison',
                    'Patient experience national comparison'], aggfunc='mean')
df5=pd.DataFrame(df5)
df5=df5.reset_index()
df5=df5.sort_values("Hospital overall rating", ascending = False).dropna()
#Ratings Correlation
g = sns.PairGrid(df5,
                 x_vars=['Hospital overall rating','Patient experience national comparison','Safety of care national comparison']
                 , y_vars=["State"],
                 size=10, aspect=.3)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=7, orient="h", palette="hls")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 5), xlabel="", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['Overall Rating','Patient Rating','Safety Rating']

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
plt.tight_layout()
sns.despine(left=True, bottom=True)
display()

# COMMAND ----------

