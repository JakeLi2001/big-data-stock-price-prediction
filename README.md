# Big Data Stock Price Prediction

**Goal**: To predict stock closing price per minute using technical indicators.

**Motivation**: I wanted to see which technical indicatosr help predict stock closing price.

## :mag_right: Data Source: [Stock Market Data - Nifty 100 Stocks (1 min) data](https://www.kaggle.com/datasets/debashis74017/stock-market-data-nifty-50-stocks-1-min-data?select=ADANIPORTS_with_indicators_.csv)

This dataset is from Kaggle and I am using version 2 of the dataset which is about 33GB.

Version 2 contains 49/50 individual stocks from the Nifty 50 Index, the Nifty 50 Index, and the Nifty Bank Index. I am only going to use the 49 individual stocks for this project. 

Features:
- Date: Date of observation
- Open: Open price of the index on a particular day
- High: High price of the index on a particular day
- Low: Low price of the index on a particular day
- Close: Close price of the index on a particular day
- And 55 technical incators...

## :open_book: Summary

1. Data Acquisition: Used Amazon EC2 to download the 33GB Kaggle dataset directly into Amazon S3.
2. Exploratory Data Analysis: Produced descriptive statistics using Amazon EC2.
3. Coding and Modeling: Built a completed machine learning pipeline using random forest regressor in AWS EMR.
4. Visualizing Results: Visualized prediction results in AWS EMR.

## :dart: Results

Model Results/Evaluation:

- R^2: 0.93
- Root Mean Square Error (RMSE): 1183
- Mean Absolute Error (MAE): 340

R^2 is the proportion of variance for a dependent variable that's explained by an independent variables (features).

Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors).

Mean Absolute Error (MAE) is the mean of absolute errors.

The top 10 features by feature importances descending are:
1. open: Open price of the index on a particular day
2. typprice: Typical price
3. high: High price of the index on a particular day
4. trima5: Triangular Moving Average of 5 close price
5. kama30: Kaufman Adaptive Moving Average of 30 close price
6. lowerband: Lowerband of Bollinger band
7. low: Low price of the index on a particular day
8. sma5: Simple moving average for 5 close price
9. ht_trendline: Hilbert Transform - Instantaneous Trendline
10. ema20: Exponential Moving Average for 20 close price

## :hammer_and_wrench: Tech Stack

**Language:** PySpark, Python

**Libraries:** boto3, zipfile, io, s3fs, pandas, numpy, pyspark.sql.types, Pipeline, StringIndexer, OneHotEncoder, VectorAssembler, RandomForestRegressor, RegressionEvaluator, chain, matplotlib, seaborn

**Tools:** Amazon Web Services (AWS) - EC2, S3, EMR

## FAQ

#### 1. What didn't I remove outliers?

I didn't remove outliers because random forest regressor is not sensitive to outliers.

#### 2. What didn't I standardize/normalize the features?

I normalized the features but saw no improve to R^2, Root Mean Square Error (RMSE), and Mean Absolute Error (MAE). Thus, I removed the code for normalization.


## Feedback

If you have any feedback, please reach out to me at LiJake2001@gmail.com.

## See my full report [here](https://github.com/JakeLi2001/big-data-stock-price-prediction/blob/main/Proejct%20Documentation.pdf)
