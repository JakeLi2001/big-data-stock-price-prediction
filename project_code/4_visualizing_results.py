# pip3 install s3fs pandas matplotlib seaborn
sc.setLogLevel("ERROR")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import s3fs

# Visualizations
# Visual 1: Overall trends time series
df = spark.read.parquet('s3a://cis4130-project-jakeli/all_stocks.parquet')
df = df.filter(df.date >= "2020-01-01 00:00:00")
df = df.select("stock", "date", "close").toPandas()

plt.figure(figsize=(15, 10))
sns.lineplot(x="date", y="close", hue="stock", data=df, legend=False).set(title='Overview of 49 Stock Trends')
plt.xlabel("Date")
plt.ylabel("Closing Price")

img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)
s3 = s3fs.S3FileSystem(anon=False)
with s3.open("s3a://cis4130-project-jakeli/visuals/stocks_time_series.png", 'wb') as f:
	f.write(img_data.getbuffer())

# Visual 2: Closing Price Distribution
df = spark.read.parquet('s3a://cis4130-project-jakeli/all_stocks.parquet')
df = df.filter(df.date >= "2020-01-01 00:00:00")
df = df.select("stock", "date", "close").toPandas()

plt.figure(figsize=(12, 8))
sns.histplot(x="close", data=df).set(title="Overall Closing Price Distribution")
plt.xlabel("Closing Price")
plt.ylabel("Frequency")

img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)
s3 = s3fs.S3FileSystem(anon=False)
with s3.open("s3a://cis4130-project-jakeli/visuals/stocks_price_distribution.png", 'wb') as f:
	f.write(img_data.getbuffer())

# Visual 3: Actual VS Prediction (49 stocks)
results_df = spark.read.parquet("s3a://cis4130-project-jakeli/rf_results.parquet").toPandas()
stocks = results_df["stock"].unique()
print(stocks)

for x in stocks:
	stock_df = results_df[results_df["stock"]==x]
	plot_title = "Actual Closing Price VS Prediction" + " (" + x + ")"
	plt.figure(figsize=(15, 10))
	sns.lineplot(x="date", y="close", data=stock_df)
	sns.lineplot(x="date", y="prediction", data=stock_df).set(title=plot_title)
	plt.xlabel("Date")
	plt.ylabel("Stock Price")
	path_name = "s3a://cis4130-project-jakeli/visuals/actual_vs_prediction" + "_" + x + ".png"
	img_data = io.BytesIO()
	plt.savefig(img_data, format='png', bbox_inches='tight')
	img_data.seek(0)
	s3 = s3fs.S3FileSystem(anon=False)
	with s3.open(path_name, 'wb') as f:
		f.write(img_data.getbuffer())

# Visual 4: Absolute Error Distribution
results_df = spark.read.parquet("s3a://cis4130-project-jakeli/rf_results.parquet").toPandas()
results_df["abs_error"] = abs(results_df["close"]-results_df["prediction"])

plt.figure(figsize=(12, 8))
sns.histplot(x="abs_error", data=results_df).set(title="Absolute Error Distribution")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")

img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)
s3 = s3fs.S3FileSystem(anon=False)
with s3.open("s3a://cis4130-project-jakeli/visuals/abs_error_distribution.png", 'wb') as f:
	f.write(img_data.getbuffer())

# Visual 5: Feature importance (Fit and transform using random forest regressor first then run the follow)
plt.figure(figsize=(18, 8))
sns.barplot(x="Feature", y="Importance", data=features_df).set(title="Feature Importance Descending") 
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, horizontalalignment='right')

img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)
s3 = s3fs.S3FileSystem(anon=False)
with s3.open("s3a://cis4130-project-jakeli/visuals/feature_importance.png", 'wb') as f:
	f.write(img_data.getbuffer())
