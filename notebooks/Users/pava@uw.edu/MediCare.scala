// Databricks notebook source
val loan=sc.textFile("dbfs:/autumn_2019/pava/project/medicare.xlsx")

// COMMAND ----------

import org.apache.spark.sql.functions._

// COMMAND ----------

loan.show()


// COMMAND ----------

