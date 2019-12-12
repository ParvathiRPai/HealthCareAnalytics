# Databricks notebook source
# MAGIC %md
# MAGIC #Health Care Analytics
# MAGIC ###Summary:
# MAGIC * Health care industry is huge and there is lot of data coming out of health care.
# MAGIC * Overall spending is $3.8 trillion per year in US health care. 
# MAGIC * The estimated waste is $765 billion dollors.
# MAGIC * Not only is cost an issue, but health care is poor. 
# MAGIC * In this project we are focusing on analysis of US health care and proving solutions that provide better health care at lower cost.
# MAGIC 
# MAGIC ### Data Set:
# MAGIC data.medicare.gov, healthcare.gov 
# MAGIC 
# MAGIC ### Tools
# MAGIC PySpark, Python (Visualizations)
# MAGIC 
# MAGIC ### By
# MAGIC Ekta Shah, Parvathi Pai

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 1: Average cost of insurance in 50 states
# MAGIC * For this analysis we used data from inpatient prospective payment system.
# MAGIC * The data includes hospital specific charges for 3000 US hospitals which covers 50 states.
# MAGIC * Average covered charge: The average charge billed to the medicare by the provider.
# MAGIC * Average total payments: The total payments made to the provider (including payments from medicare as well as the co-payments and deductibles paid by the beneficiary)
# MAGIC * Average Medicare Payments: The average payment just from medicare.
# MAGIC * Variation in insurance cost based on age.

# COMMAND ----------

# MAGIC %scala
# MAGIC val ippsdf =  spark.read.option("inferSchema","true").option("header","true").csv("dbfs:/autumn_2019/ektashah/ProjectDataSet/IPPS_2017.csv")
# MAGIC import org.apache.spark.sql.functions._
# MAGIC val a_ippdf_1 = ippsdf.groupBy(($"Provider State") as "ProviderState" ).agg(avg($"Average Medicare Payments") as "AverageMedicarePayment",avg($"Average Total Payments") as "AverageTotalPayment")
# MAGIC val a_ippsdf_2 = ippsdf.groupBy(($"DRG Definition") as "DRG Definition", ($"Provider Name") as "ProviderName", ($"Provider State") as "ProviderState" ).agg(avg($"Average Medicare Payments") as "AverageMedicarePayment",avg($"Average Total Payments") as "AverageTotalPayment")

# COMMAND ----------

# MAGIC %scala
# MAGIC display(a_ippdf_1)

# COMMAND ----------

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df_average_insurance=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/Insurance.csv')
df_average_insurance.describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC _For the same count of subscribers their is variation of insurance with respect to age and children._

# COMMAND ----------

Statelist = df_average_insurance['State'].unique()
Statelist = np.sort(Statelist)
plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")
sns.boxplot(x="State", y="Premium Child",data=df_average_insurance, order=Statelist)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC _Insurance premium for children is highest fluctuated in VA._

# COMMAND ----------

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df_va=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/Rate.csv')
na_values = ['NaN', 'N/A', '0', '0.01', '9999', '9999.99', '999999']
df_va = pd.read_csv("/dbfs/autumn_2019/pava/ProjectData/Rate.csv", na_values=na_values, usecols=['BusinessYear', 'StateCode', 'PlanId', 'RatingAreaId','Tobacco', 'Age', 'IndividualRate','IndividualTobaccoRate','Couple', 'PrimarySubscriberAndOneDependent', 'PrimarySubscriberAndTwoDependents', 'PrimarySubscriberAndThreeOrMoreDependents', 'CoupleAndOneDependent', 'CoupleAndTwoDependents', 'CoupleAndThreeOrMoreDependents'])
df_va = df_va.drop_duplicates().reset_index(drop=True)
sns.boxplot(x="BusinessYear", y="IndividualRate", data=df_va)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC _But average insurance rate is same in all three years with some outliers in 2014 _

# COMMAND ----------

Statelist = df_va['StateCode'].unique()
Statelist = np.sort(Statelist)
plt.figure(figsize=(15, 6))
sns.boxplot(x="StateCode", y="IndividualRate", data=df_va, order=Statelist)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC Adult insurance in VA is comparable with other states. Only 75 percentile of poplution pay more for child insurance in VA

# COMMAND ----------

fig = plt.figure(figsize=(15, 6))
sns.set_style("whitegrid")

fig.suptitle('Rates offered through Healthcare.gov in 2014', fontsize=14)
df2014 = df_va[df_va['BusinessYear'].isin([2014])].copy()
ax = sns.boxplot(x="Age", y="IndividualRate", data=df2014, linewidth=1.0, fliersize=2.0)
ax.set_ylabel("Monthly rate in USD")
list = df2014['Age'].isin(['Family Option'])
df2014_va_wofamily = df2014[~list]
age_labels = df2014_va_wofamily['Age'].unique()
age_labels = ['65+' if x=='65 and over' else x for x in age_labels]  #replace label '65 and over' with '65+' for plot labels
len(age_labels), age_labels
xticks = np.arange(46)
ax.xaxis.set_ticks(xticks)
ax.set_xticklabels(age_labels)

plt.savefig('Virginia_rates_by_age.png', bbox_inches='tight', dpi=150)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC The rate of premium insurance increases with age 

# COMMAND ----------

median=pd.DataFrame(df_average_insurance.groupby('Plan ID - Standard Component',sort=False)['Premium Adult Individual Age 21','Premium Adult Individual Age 27','Premium Adult Individual Age 30','Premium Adult Individual Age 40','Premium Adult Individual Age 50','Premium Adult Individual Age 60'].median())
median=median.rename(columns={'Premium Adult Individual Age 21':'median_Premium Adult Individual Age 21','Premium Adult Individual Age 27':'median_Premium Adult Individual Age 27','Premium Adult Individual Age 30':'median_Premium Adult Individual Age 30','Premium Adult Individual Age 40':'median_Premium Adult Individual Age 40','Premium Adult Individual Age 50':'median_Premium Adult Individual Age 50','Premium Adult Individual Age 60':'median_Premium Adult Individual Age 60'})
median.head(3)

# COMMAND ----------

# MAGIC %md 
# MAGIC _ The insurance cost increases with age by 5 to 10%._

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 2: Average medical cost based by BMI and Gender
# MAGIC * In this analysis we calculate medical charges per region i.e North, South, East, West.
# MAGIC * Fluctuations of medical charges for smokers and non-smokers.
# MAGIC * Analyze how medical charges varies with increasing BMI and age for smoker and non-smoker
# MAGIC * Analyze average medical cost based on body type and gender
# MAGIC * Medical charges analysis by gender and age groups

# COMMAND ----------

# MAGIC %scala
# MAGIC val insurancedf = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load("dbfs:/autumn_2019/pava/ProjectData/insurance.csv")
# MAGIC import org.apache.spark.sql.functions._
# MAGIC val a_insurancedf_1 = insurancedf.withColumn("BMI",
# MAGIC        when(col("bmi") < 18.5, "Underweight")
# MAGIC       .when(col("bmi") >= 18.5 && col("bmi") <= 24.9, "Normal")
# MAGIC       .otherwise("Overweight"))
# MAGIC       .groupBy("sex","BMI").agg(avg($"charges") as "AvgMedicalCost").orderBy(desc("AvgMedicalCost"))
# MAGIC val a_insurancedf_2 = insurancedf.withColumn("AgeGroup",
# MAGIC        when(col("age") <= 20, "0-20")
# MAGIC       .when(col("age") > 20 && col("age") <= 40, "20-40")
# MAGIC       .when(col("age") > 40 && col("age") <= 60, "40-60") 
# MAGIC       .when(col("age") > 60 && col("age") <= 80, "60-80")                          
# MAGIC       .otherwise(">80"))
# MAGIC       .groupBy("sex","AgeGroup").agg(avg($"charges") as "AvgMedicalCost").orderBy(desc("AvgMedicalCost"))

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
df_smoker_nonsmoker=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/insurance_smoker.csv')
sns.boxplot(x=df_smoker_nonsmoker.region,y=df_smoker_nonsmoker.charges, data=df_smoker_nonsmoker)
plt.title("Medical Charges per region")
plt.show()
display()

# COMMAND ----------

sns.boxplot(x=df_smoker_nonsmoker.smoker, y=df_smoker_nonsmoker.charges, data=df_smoker_nonsmoker)
plt.title("Medical charges of smoker and non-smoker")
display()

# COMMAND ----------

sns.lmplot(x="bmi", y="charges", hue='smoker',data=df_smoker_nonsmoker)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####* _Medical charges increases in case of smoker with the increasing BMI. But in case of non-smokers the increasing BMI doesn't have large impact on the medical charges._ 

# COMMAND ----------

sns.lmplot(x='age', y='charges', hue='smoker',data=df_smoker_nonsmoker,palette='inferno_r')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The cost of treatment increases with age for both smokers and non-smokers. The above graph also depicts the variation of cost between habitual and recreational smokers._ 

# COMMAND ----------

# MAGIC %scala
# MAGIC display(a_insurancedf_1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The cost of treatment is higher for overweight males compare to overweight females, whereas the of the medical cost is higher for underweight females compare to underweight males_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 3: Average medicare cost based on Provider Type
# MAGIC * In this analysis we are using Medicare Provider Utilization and Payment Data of year 2017 to do our analysis.
# MAGIC * This analysis focus on finding out the highest costing provider type in USA in 2017.
# MAGIC * Here we are also analyzing the % medicare cost of each provider type in 2017.

# COMMAND ----------

# MAGIC %scala
# MAGIC val pufdf =  spark.read.option("inferSchema","true").option("header","true").csv("dbfs:/autumn_2019/ektashah/ProjectDataSet/PUF_CY2017.csv")
# MAGIC import org.apache.spark.sql.functions._
# MAGIC val a_pufdf_1 = pufdf.filter($"Country Code of the Provider" === "US").groupBy(($"Provider Type") as "ProviderType").agg(sum($"Average Medicare Payment Amount") as "MedicarePayment").orderBy(desc("MedicarePayment"))

# COMMAND ----------

# MAGIC %scala
# MAGIC display(a_pufdf_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 4: Average Medicare Payment Amount Comparison based on Provider Type
# MAGIC * As per our last analysis Internal Medicine is the highest costing provider type, so here we are further analysing it cost in each states.

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._
# MAGIC val a_pufdf_2 = pufdf.filter($"Country Code of the Provider" === "US" && $"Provider Type" === "Internal Medicine" ).groupBy("State Code of the Provider").agg(sum($"Average Medicare Payment Amount") as "MedicarePayment").orderBy(desc("MedicarePayment"))

# COMMAND ----------

# MAGIC %scala
# MAGIC display(a_pufdf_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The cost of Medicare is higher in CA and in NY for Internal Medicine Provider_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 5: Comparison between Average Medicare Cost from 2014 to 2017 
# MAGIC * In this analysis we are comparing the Average Medicare Cost of 2014 with 2017 and analyzing the cost increment in 4 years.

# COMMAND ----------

# MAGIC %scala
# MAGIC val puf_2014 =  spark.read.option("inferSchema","true").option("header","true").csv("dbfs:/autumn_2019/ektashah/ProjectDataSet/PUF_CY2014.csv")
# MAGIC val puf_2015 =  spark.read.option("inferSchema","true").option("header","true").csv("dbfs:/autumn_2019/ektashah/ProjectDataSet/PUF_CY2015.csv")
# MAGIC val puf_2016 =  spark.read.option("inferSchema","true").option("header","true").csv("dbfs:/autumn_2019/ektashah/ProjectDataSet/PUF_CY2016.csv")
# MAGIC import org.apache.spark.sql.functions._
# MAGIC val a2014 = puf_2014.filter($"Country Code of the Provider" === "US").groupBy(($"Provider Type of the Provider") as "ProviderType_2014").agg(sum($"Average Medicare Payment Amount") as "MedicarePayment_2014").orderBy(desc("MedicarePayment_2014"))
# MAGIC val a2015 = puf_2015.filter($"Country Code of the Provider" === "US").groupBy(($"Provider Type") as "ProviderType_2015").agg(sum($"Average Medicare Payment Amount") as "MedicarePayment_2015").orderBy(desc("MedicarePayment_2015"))
# MAGIC val a2016 = puf_2016.filter($"Country Code of the Provider" === "US").groupBy(($"Provider Type") as "ProviderType_2016").agg(sum($"Average Medicare Payment Amount") as "MedicarePayment_2016").orderBy(desc("MedicarePayment_2016"))
# MAGIC val a2017 = pufdf.filter($"Country Code of the Provider" === "US").groupBy(($"Provider Type") as "ProviderType_2017").agg(sum($"Average Medicare Payment Amount") as "MedicarePayment_2017").orderBy(desc("MedicarePayment_2017"))
# MAGIC val join2016  =  a2017.join(a2016, $"ProviderType_2017" === $"ProviderType_2016").select($"ProviderType_2017", $"MedicarePayment_2017", $"MedicarePayment_2016")
# MAGIC val join2015  =  join2016.join(a2015, $"ProviderType_2017" === $"ProviderType_2015").select($"ProviderType_2017", $"MedicarePayment_2017", $"MedicarePayment_2016", $"MedicarePayment_2015")
# MAGIC val join2014  =  join2015.join(a2014, $"ProviderType_2017" === $"ProviderType_2014").select($"ProviderType_2017", $"MedicarePayment_2017", $"MedicarePayment_2016",$"MedicarePayment_2015",$"MedicarePayment_2014")

# COMMAND ----------

# MAGIC %scala
# MAGIC display(join2014)

# COMMAND ----------

# MAGIC %md
# MAGIC #### _The overall Medical Cost has increased for all the Provider Types from 2014 to 2017_

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
df_cost=pd.read_csv('/dbfs/autumn_2019/pava/ProjectData/IPPS_Data_Clean.csv')
df_cost.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Billed charges vary from hopital to hospital for the same procedure.

# COMMAND ----------

nationalmedian = pd.DataFrame(df_cost.groupby('drg_id',sort=False)['average_covered_charges','average_total_payments','average_medicare_payments'].median()).reset_index()
nationalmedian = nationalmedian.rename(columns={'average_covered_charges':'median_covered_charges', 'average_total_payments':'median_total_payments','average_medicare_payments':'median_medicare_payments' })


# COMMAND ----------

costoftreatmentperstate=df_cost.groupby(['drg_id','provider_state']).size()
costoftreatmentperstate =costoftreatmentperstate.unstack('provider_state').fillna(0)
costoftreatmentperstate.sum().plot(ylim=0,kind='bar',figsize=(25,8),fontsize=16);
plt.ylabel('ID- 39 treatment procedure charges variation by state',fontsize=16)
display()


# COMMAND ----------

# MAGIC %md 
# MAGIC For the same treatment, variation of medical charges with respect to state

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis 6: Hospital rating
# MAGIC * Hospital ownership (Government, private, non-profit) and their average rating
# MAGIC * The average rating is based on hospital overall rating, saftey of care national rating, patient experience. 

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

# MAGIC %md
# MAGIC ## Analysis 7: Hospital Readmission
# MAGIC * In this analysis we try to find if there is any relation between number of discharges and readmission ratio.
# MAGIC * Readmission ratio is the ratio of predicted rate/Expected rate.
# MAGIC * predicated rate - 30 day readmission for heart attack (AMI), heart failure (HF), pneumonia, chronic obstructive pulmonary disease (COPD), hip/knee replacement (THA/TKA), and coronary artery bypass graft surgery (CABG).

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy as sp
df_readmission=pd.read_csv("/dbfs/autumn_2019/pava/ProjectData/cms_readmissions.csv")
df_readmission.head()

# COMMAND ----------

# Analysis:Remove missing portion of data
clean_df=df_readmission[df_readmission['Number of Discharges'] != 'Not Available']
clean_df.loc[:,'Number of Discharges']=clean_df['Number of Discharges'].astype(int)
clean_df=clean_df.sort_values('Number of Discharges')
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

# MAGIC %md
# MAGIC * Rate of readmission is trending down with increase in number of discharges.
# MAGIC * With lower number of discharges there is excess rate of readmission.
# MAGIC * With higher number of discharges lower rate of readmissions.
# MAGIC 
# MAGIC ####Statistics:
# MAGIC * In hospitals with discharges <100, mean excess readmission rate is 1.023 and 63% have excess readmission rate greater than 1.
# MAGIC * In hospitals with discharges>1000, mean excess readmission rate is 0.978 and 44% have excess readmission rate greater than 1.
# MAGIC 
# MAGIC ####Conclusion: 
# MAGIC * There is a significant correlation between hospital capacity and readmission rates.
# MAGIC * Smaller hospitals/facilities may be lacking necessary resources to ensure quality care this may lead to readmission rate greater than 1.

# COMMAND ----------

x=pd.DataFrame(x)
y=pd.DataFrame(y)
df = pd.concat([x,y], axis=1)
df.columns = ['discharges', 'excess_readmissions_ratio']
df.head()


# COMMAND ----------

sns.jointplot('discharges','excess_readmissions_ratio', data=df, kind='reg' )
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ####_Overall, rate of readmission is slightly trending down with the increasing number of discharges._

# COMMAND ----------

red=df[df.discharges <= 350]
red=red[red['excess_readmissions_ratio']>=1.15]
red = red[red['excess_readmissions_ratio'] <= 2.00]
sns.jointplot('discharges', 'excess_readmissions_ratio', data=red, kind='reg', color="r")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ####_With lower number of discharges, there is a greater incidence of excess rate of readmissions_

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Readmissions by state:
# MAGIC * Top 10 states with low and high excess readmission rates

# COMMAND ----------

## Hospital readmission based on state
x=list(clean_df['State'][81:-3])
y=list(clean_df['Excess Readmission Ratio'][81:-3])
x=pd.DataFrame(x)
y=pd.DataFrame(y)

df=pd.concat([x,y], axis=1)
df.columns=['state', 'excess readmission ratio']
df=df.groupby('state').mean().reset_index()
lo_excess=df.sort_values(by='excess readmission ratio').head(10)
hi_excess=df.sort_values(by='excess readmission ratio').tail(10)
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

# Data for South Dakota
sd=df[df.state == 'SD']
sd = sd['excess_readmissions']
# sd.describe()

# Data for DC (District of Columbia)

dc = df[df.state == 'DC']
dc = dc['excess_readmissions']
#dc.describe()
sd=df[df.state == 'SD']
sd = sd[['excess_readmissions']]
sd.columns = ['South Dakota Excess Readmission Rate']

dc = df[df.state == 'DC']
dc = dc[['excess_readmissions']]
dc.columns = ['            District of Columbia Excess Readmission Rate']

fig, axs = plt.subplots(ncols=2, sharey=True)
sns.violinplot(data=sd, ax=axs[0], color='purple')
sns.violinplot(data=dc, ax=axs[1], color='blue')
plt.tight_layout()
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Violin plot is a combination of box plot and density plot. 
# MAGIC * From the above analysis South Dakota(SD) had lower readmission ratio and District of Columbia (DC) had the greatest readmission ratio. 
# MAGIC * From the violin plot it can be seen that the SD mean readmission ratio is around 0.9 and the distribution is uniform.
# MAGIC * District of Columbia mean readmission ratio is 1.1 and their is a slight fluctuation in the distribution of data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result:
# MAGIC * If people cultivate healthy life style then medical cost reduces :) 
# MAGIC * The south east region has higher fluctuations in medical cost.
# MAGIC * For smokers the medical cost increases indisputably with BMI and age. 
# MAGIC * Its better to prefer hospital with higher rate of discharge and lower readmission ratio.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="http://mrkinsella.weebly.com/uploads/3/7/0/9/37098191/182427707.jpg?434">

# COMMAND ----------


