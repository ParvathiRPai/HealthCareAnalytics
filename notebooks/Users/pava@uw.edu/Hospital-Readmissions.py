# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy as sp

# COMMAND ----------

df=pd.read_csv("/dbfs/autumn_2019/pava/ProjectData/cms_readmissions.csv")
df.head()

# COMMAND ----------

# Analysis:Remove missing portion of data
clean_df=df[df['Number of Discharges'] != 'Not Available']
clean_df.loc[:,'Number of Discharges']=clean_df['Number of Discharges'].astype(int)
clean_df=clean_df.sort_values('Number of Discharges')


# COMMAND ----------

# Scatter plot for number of discharges versus rate of readmissions
x=[a for a in clean_df['Number of Discharges'][81:-3]]
y=list(clean_df['Excess Readmission Ratio'][81:-3])
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(x, y,alpha=0.2)

ax.fill_between([0,350], 1.15, 2, facecolor='red', alpha = .15, interpolate=True)
ax.fill_between([800,2500], .5, .95, facecolor='green', alpha = .15, interpolate=True)

ax.set_xlim([0, max(x)])
ax.set_xlabel('Number of discharges', fontsize=12)
ax.set_ylabel('Excess rate of readmissions ratio', fontsize=12)
ax.set_title('Scatterplot of number of discharges vs. excess rate of readmissions ratio', fontsize=14)

ax.grid(True)
fig.tight_layout()
display()

# COMMAND ----------

# Rate of readmission is trending down with increase in number of discharges.
# With lower number of discharges there is excess rate of readmission.
# with higher number of discharges lower rate of readmissions.
x=pd.DataFrame(x)
y=pd.DataFrame(y)
df = pd.concat([x,y], axis=1)
df.columns = ['discharges', 'excess_readmissions_ratio']
df.head()


# COMMAND ----------

sns.jointplot('discharges','excess_readmissions_ratio', data=df, kind='reg' )
display()

# COMMAND ----------

# With lower number of discharges, there is a greater incidence of excess rate of readmission
red=df[df.discharges <= 350]
red=red[red['excess_readmissions_ratio']>=1.15]
red = red[red['excess_readmissions_ratio'] <= 2.00]
sns.jointplot('discharges', 'excess_readmissions_ratio', data=red, kind='reg', color="r")
display()

# COMMAND ----------

## Hospital readmission based on state
x=list(clean_df['State'][81:-3])
y=list(clean_df['Excess Readmission Ratio'][81:-3])
x=pd.DataFrame(x)
y=pd.DataFrame(y)

df=pd.concat([x,y], axis=1)
df.columns=['state', 'excess readmission ratio']
df=df.groupby('state').mean().reset_index()
df.head()

# COMMAND ----------

lo_excess=df.sort_values(by='excess readmission ratio').head(10)
hi_excess=df.sort_values(by='excess readmission ratio').tail(10)

# COMMAND ----------

sns.barplot('state','excess readmission ratio', data=lo_excess)
plt.title("Top 10 states with low excess readmission rates")
display()


# COMMAND ----------

sns.barplot('state','excess readmission ratio', data=hi_excess)
plt.title('Top 10 states with high excess readmission rates')
display()

# COMMAND ----------

x=list(clean_df['State'][81:-3])
y=list(clean_df['Excess Readmission Ratio'][81:-3])
x = pd.DataFrame(x)
y = pd.DataFrame(y)

df = pd.concat([x,y], axis=1)
df.columns = ['state', 'excess_readmissions']
df.head()

# COMMAND ----------

# Data for SD
sd=df[df.state == 'SD']
sd = sd['excess_readmissions']
sd.describe()

# COMMAND ----------

# Data for DC (District of Columbia)

dc = df[df.state == 'DC']
dc = dc['excess_readmissions']
dc.describe()

# COMMAND ----------

sd=df[df.state == 'SD']
sd = sd[['excess_readmissions']]
sd.columns = ['South Dakota Excess Readmission Rate']

dc = df[df.state == 'DC']
dc = dc[['excess_readmissions']]
dc.columns = ['             District of Columbia Excess Readmission Rate']

fig, axs = plt.subplots(ncols=2, sharey=True)
sns.violinplot(data=sd, ax=axs[0], color='#CCCCFF')
sns.violinplot(data=dc, ax=axs[1], color='#FFCCCC')
display()

# COMMAND ----------

