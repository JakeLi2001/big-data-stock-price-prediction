# import libraries
import s3fs
import pandas as pd
import boto3

client = boto3.client('s3')
bucket = 'cis4130-project-jakeli'
resp = client.list_objects_v2(Bucket = bucket)

# get stock names from files names
stock_names = []
for i in resp['Contents']:
	if ('.csv' in i['Key']):
		file_name = i['Key']
		stock_name = file_name.split('_', 1)[0]
		stock_names.append(stock_name)

# merge CSV files
df = pd.DataFrame()
for i in stock_names:
	path = "s3://" + bucket + "/" + i + "_with_indicators_.csv"
	df_temp = pd.read_csv(path)
	df_temp['stock'] = i
	cols = df_temp.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	df_temp = df_temp[cols]
	df = df.append(df_temp)

# export dataframe to S3
df.to_csv('s3://cis4130-project-jakeli/all_stocks.csv', index=False)

# Descriptive Statistics
df.shape
# column names
df.columns
# sum of null values
df.isnull().sum().sum()
# sum of duplicated rows
df.duplicated().sum().sum()
# earliest and latest dates
min(df['date'])
max(df['date'])
# dataframe information
df.info()
# generate descriptive statistics for open, high, low, close, and volume column
round(df[["open", "high", "low", "close", "volume"]].describe())
