# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

df=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/IPPS_Data_Clean.csv',dtype={'provider_id':str,'provider_zip_code':str, 'drg_id':str})
df.head(3)

# COMMAND ----------

nationalmedian = pd.DataFrame(df.groupby('drg_id',sort=False)['average_covered_charges','average_total_payments', \
                                                       'average_medicare_payments'].median()).reset_index()
nationalmedian = nationalmedian.rename(columns={'average_covered_charges':'median_covered_charges', \
                                  'average_total_payments':'median_total_payments', \
                                  'average_medicare_payments':'median_medicare_payments' })
nationalmedian.head(3)

# COMMAND ----------

nationalmedian.set_index('drg_id').plot(kind='bar',figsize = (25,12),color=('r','b','g'),fontsize=14)
display()

# COMMAND ----------

costoftreatmentperstate = df.groupby(['drg_id','provider_state']).size()
costoftreatmentperstate = costoftreatmentperstate.unstack('provider_state').fillna(0)
costoftreatmentperstate.head()


# COMMAND ----------

costoftreatmentperstate.sum(1).plot(ylim=0,kind='bar',figsize=(25,8),fontsize=14);
display()

# COMMAND ----------

costoftreatmentperstate.sum().plot(ylim=0,kind='bar',figsize=(25,8),fontsize=16);
display()


# COMMAND ----------

