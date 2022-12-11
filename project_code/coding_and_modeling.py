sc.setLogLevel("ERROR")
# Import libraries
import pandas as pd
import numpy as np
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from itertools import chain

""" Run below once
# Define schema
schema = StructType([
	StructField("stock", StringType()),
	StructField("date", TimestampType()),
	StructField("open", DoubleType()),
	StructField("high", DoubleType()),
	StructField("low", DoubleType()),
	StructField("close", DoubleType()),
	StructField("volumn", DoubleType()),
	StructField("sma5", DoubleType()),
	StructField("sma10", DoubleType()),
	StructField("sma15", DoubleType()),
	StructField("sma20", DoubleType()),
	StructField("ema5", DoubleType()),
	StructField("ema10", DoubleType()),
	StructField("ema15", DoubleType()),
	StructField("ema20", DoubleType()),
	StructField("upperband", DoubleType()),
	StructField("middleband", DoubleType()),
	StructField("lowerband", DoubleType()),
	StructField("HT_Trendline", DoubleType()),
	StructField("KAMA10", DoubleType()),
	StructField("KAMA20", DoubleType()),
	StructField("KAMA30", DoubleType()),
	StructField("SAR", DoubleType()),
	StructField("TRIMA5", DoubleType()),
	StructField("TRIMA10", DoubleType()),
	StructField("TRIMA20", DoubleType()),
	StructField("ADX5", DoubleType()),
	StructField("ADX10", DoubleType()),
	StructField("ADX20", DoubleType()),
	StructField("APO", DoubleType()),
	StructField("CCI5", DoubleType()),
	StructField("CCI10", DoubleType()),
	StructField("CCI15", DoubleType()),
	StructField("macd510", DoubleType()),
	StructField("macd520", DoubleType()),
	StructField("macd1020", DoubleType()),
	StructField("macd1520", DoubleType()),
	StructField("macd1226", DoubleType()),
	StructField("MOM10", DoubleType()),
	StructField("MOM15", DoubleType()),
	StructField("MOM20", DoubleType()),
	StructField("ROC5", DoubleType()),
	StructField("ROC10", DoubleType()),
	StructField("ROC20", DoubleType()),
	StructField("PRO", DoubleType()),
	StructField("RSI14", DoubleType()),
	StructField("RSI8", DoubleType()),
	StructField("slowk", DoubleType()),
	StructField("slowd", DoubleType()),
	StructField("fastk", DoubleType()),
	StructField("fastd", DoubleType()),
	StructField("fastksr", DoubleType()),
	StructField("fastdsr", DoubleType()),
	StructField("ULTOSC", DoubleType()),
	StructField("WILLR", DoubleType()),
	StructField("ATR", DoubleType()),
	StructField("Trange", DoubleType()),
	StructField("TYPPRICE", DoubleType()),
	StructField("HT_DCPERIOD", DoubleType()),
	StructField("BETA", DoubleType())])
# Read csv from S3
df = spark.read.csv('s3a://cis4130-project-jakeli/all_stocks.csv', header=True, schema=schema)
df.printSchema()
# Change all columns names to lowercase
for col in df.columns:
    df = df.withColumnRenamed(col, col.lower())
# Export dataframe as parquet file
df.write.parquet("s3a://cis4130-project-jakeli/all_stocks.parquet")
"""

### Start from here 1
# df = spark.read.parquet('s3a://cis4130-project-jakeli/all_stocks.parquet')

# Building pipeline
selected_features = df.columns
del selected_features[0:2]
selected_features.remove("close")
print(selected_features)

assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
pipeline = Pipeline(stages=[assembler])
transformed_df = pipeline.fit(df).transform(df)

transformed_df = transformed_df["stock", "date", "close", "features"]
transformed_df.write.mode("overwrite").parquet("s3a://cis4130-project-jakeli/all_stocks_transformed.parquet")

### Start from here 2
# transformed_df = spark.read.parquet("s3a://cis4130-project-jakeli/all_stocks_transformed.parquet")

# Random Forest Regressor Model
max_date = transformed_df.agg({"date": "max"}).collect()[0][0]
min_date = transformed_df.agg({"date": "min"}).collect()[0][0]
seventy_percent_threshold = (max_date - min_date)*0.7
split_date = min_date + seventy_percent_threshold
print(split_date)

train_df = transformed_df.where(transformed_df["date"] < split_date)
test_df = transformed_df.where(transformed_df["date"] >= split_date)

rf = RandomForestRegressor(labelCol="close", featuresCol="features", seed=2)
model = rf.fit(train_df)
prediction = model.transform(test_df)

# Calculate RMSE, R^2, and MAE score
evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction")
r2 = evaluator.evaluate(prediction, {evaluator.metricName: "r2"})
rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(prediction, {evaluator.metricName: "mae"})
print("The R^2 for using all features is:", round(r2, 2))
print("The RMSE for using all features is:", round(rmse, 2))
print("The MAE for using all features is:", round(mae, 2))

rf_results = prediction.select("stock", "date", "close", "prediction")
rf_results.write.mode("overwrite").parquet('s3a://cis4130-project-jakeli/rf_results.parquet')

# Run model again using only the top 10 features
attrs = sorted((attr["idx"], attr["name"]) for attr in (chain(*transformed_df.schema["features"].metadata["ml_attr"]["attrs"].values()))) 
feature_list = [(name, round(model.featureImportances[idx],5)) for idx, name in attrs if model.featureImportances[idx]]
features_df = pd.DataFrame(feature_list, columns=["feature", "importance"]).sort_values(by="importance", ascending=False)
features_df.head(10)
top_10_features = features_df["feature"][0:10].tolist()

df = spark.read.parquet('s3a://cis4130-project-jakeli/all_stocks.parquet')

assembler = VectorAssembler(inputCols=top_10_features, outputCol="features")
pipeline = Pipeline(stages=[assembler])
transformed_df2 = pipeline.fit(df).transform(df)

transformed_df2 = transformed_df["stock", "date", "close", "features"]
transformed_df2.write.mode("overwrite").parquet("s3a://cis4130-project-jakeli/all_stocks_transformed_2.parquet")

transformed_df2 = spark.read.parquet("s3a://cis4130-project-jakeli/all_stocks_transformed_2.parquet")
max_date = transformed_df2.agg({"date": "max"}).collect()[0][0]
min_date = transformed_df2.agg({"date": "min"}).collect()[0][0]
seventy_percent_threshold = (max_date - min_date)*0.7
split_date = min_date + seventy_percent_threshold
print(split_date)
train_df2 = transformed_df2.where(transformed_df2["date"] < split_date)
test_df2 = transformed_df2.where(transformed_df2["date"] >= split_date)

rf = RandomForestRegressor(labelCol="close", featuresCol="features", seed=2)
model = rf.fit(train_df2)
prediction = model.transform(test_df2)

# Calculate RMSE, R^2, and MAE score
evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction")
r2 = evaluator.evaluate(prediction, {evaluator.metricName: "r2"})
rmse = evaluator.evaluate(prediction, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(prediction, {evaluator.metricName: "mae"})
print("The R^2 for using top 10 features only is:", round(r2, 2))
print("The RMSE for using top 10 features only is:", round(rmse, 2))
print("The MAE for using top 10 features only is:", round(mae, 2))

rf_results2 = prediction.select("stock", "date", "close", "prediction")
rf_results2.write.mode("overwrite").parquet('s3a://cis4130-project-jakeli/rf_results_2.parquet')