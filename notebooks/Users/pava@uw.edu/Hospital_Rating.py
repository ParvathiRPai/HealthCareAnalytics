# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy as sp

# COMMAND ----------

df_info=pd.read_csv("/dbfs/autumn_2019/pava/ProjectData/HospInfo.csv")

# COMMAND ----------

df_ratings=df_info.drop(['Address','ZIP Code','Phone Number','Emergency Services',
'Patient experience national comparison footnote','Efficient use of medical imaging national comparison footnote',
'Hospital overall rating footnote','Mortality national comparison footnote','Mortality national comparison footnote',
'Safety of care national comparison footnote','Readmission national comparison footnote',
'Patient experience national comparison footnote','Effectiveness of care national comparison footnote',
'Timeliness of care national comparison footnote','Timeliness of care national comparison footnote',
'Efficient use of medical imaging national comparison footnote','Meets criteria for meaningful use of EHRs',
'Provider ID'], axis=1)
df_ratings.head()

# COMMAND ----------

df1=df_ratings.replace('Below the National average',1)
df2=df1.replace('Same as the National average',2)
df3=df2.replace('Above the National average',3)

# COMMAND ----------

df3['Hospital overall rating'] = df3['Hospital overall rating'].convert_objects(convert_numeric=True)
sns.barplot(x='Hospital overall rating', y='Hospital Ownership', data=df3, palette="deep")
plt.title('Hospital Ownership & Average Rating',x=.3, y=1.05)
plt.xlabel('Rating')
plt.ylabel('Ownership Category')
plt.subplots_adjust(right=.75)
display()

# COMMAND ----------

