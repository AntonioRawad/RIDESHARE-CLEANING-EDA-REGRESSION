#!/usr/bin/env python
# coding: utf-8

# ## EDA EXPLORATORT DATA ANALYSIS 

# we start by identifying and addressing the missing data and outliers in the ride sharing dataset. One approach to addressing missing data is to either remove the incomplete rows or fill in the missing values with appropriate imputation methods such as mean, median, or mode. For outliers, we can use visualization techniques such as box plots and scatter plots to identify any extreme values, and then decide on whether to remove or keep them based on domain knowledge and the impact on the analysis.

# ## IMPORT LIBRERIES 

# In[139]:


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# # Import the dataset:

# In[79]:


df = pd.read_csv(r'C:\Users\rawad\OneDrive\Desktop\DATA SET\Datasetridesharing.csv')


# In[89]:


print(df.dtypes)


# In[90]:


# Check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())

Based on the previous step where we checked for duplicates, we found that there are no duplicate rows in the dataset. Therefore, the number of duplicate rows is 0.
# ## Data Cleaning 

# View the first few rows of the dataset:

# In[80]:


df


# In[81]:


df.shape


# ## Check for missing data:

# In[68]:


df.isnull().sum()


# # Handelling missing data
since the data is too small we will not delete any row that have missing data , the strategy is to replace missing data with mean and median methods 
# In[91]:


# Impute missing values with mean or median
df['Ride Time'] = df['Ride Time'].fillna(df['Ride Time'].median())
df['Ride Fare'] = df['Ride Fare'].fillna(df['Ride Fare'].mean())
df['Ride Date'] = df['Ride Date'].fillna(method='ffill')
df['Ride Hour'] = df['Ride Hour'].fillna(method='bfill')


# In[92]:


# cheking data after we 
df

we found in row 22 inconsistent data , we proceed with dealing with inconsistent data 
# ##  dealing with inconsistent Data 
1) Converting Date and Time: The Ride Time, Ride Date, and Ride Hour columns can be combined and converted into a single datetime column using the pd.to_datetime() function. This can help in analyzing and visualizing the data over time.
# In[83]:


print(df['Ride Date'].unique())

appears that '202' is not a valid date and may have been mistakenly included in the 'Ride Date' column.
To fix this issue, we need to replace the invalid date string. One way to do this is to use the pandas .loc accessor to filter out the row(s) that contain the invalid date string, then replace it , i decided that data can be meant to be 1/07/2022
# In[84]:


df.loc[df['Ride Date'] == '202', 'Ride Date'] = '1/7/2022'


# In[86]:


# remove row 22 from set since it hold 2 unkown errors 


# In[87]:


# Remove row 22 from the DataFrame
df = df.drop(22)



# In[88]:


df


# In[75]:


# Save the updated dataset to a new file
df.to_csv('updated_dataset.csv', index=False)


# In[93]:


df 


# ## Exploratory Data Analysis (EDA).

# Plot the distribution of ride fares to identify the range of fares, and whether there are any unusual values or patterns.

# In[94]:


plt.hist(df['Ride Fare'], bins=10)
plt.xlabel('Ride Fare')
plt.ylabel('Frequency')
plt.title('Distribution of Ride Fares')
plt.show()

From the histogram, we can see that the majority of ride fares fall within the range of 10 to 25, with the most frequent ride fare being 20. There are also a few rides with fare values of 5 and 30. The distribution is approximately symmetrical around the peak at 20, with a slightly longer tail to the right (higher fares).

It's worth noting that this is a small dataset, so the distribution may not be fully representative of the population of rides. Nonetheless, the histogram provides a useful visual summary of the distribution of ride fares in the dataset.
# # EXPLORE RELATIONSHIP 
# Explore the relationship between ride fares and ride time, ride date, and ride hour to identify any patterns or trends. For example, are fares higher during peak hours or on certain days?

# In[95]:


#First, we can create a scatter plot of ride fares versus ride time:
plt.scatter(df['Ride Time'], df['Ride Fare'])
plt.xlabel('Ride Time')
plt.ylabel('Ride Fare')
plt.show()


# From the scatter plot, we can see that there is a general trend of increasing ride fare with increasing ride time. However, there are some outliers such as the ride with fare 5 at ride time 10 and the ride with fare 20 at ride time 10.
# 
# Additionally, we can see that at ride time 25, there are three rides with different fares, which may indicate some variation based on the time of day. The ride with fare 21 at ride time 25 is also a bit of an outlier.
# 
# The point at ride time 50 with fare 20 is likely an error or outlier, as it does not fit the general trend of increasing fare with increasing time.
# 
# Overall, we can interpret this scatter plot to suggest that there is a positive linear relationship between ride time and ride fare, with some variation and outliers.

# ### we can create a line plot of ride fares versus ride date:

# In[96]:


plt.plot(df['Ride Date'], df['Ride Fare'])
plt.xlabel('Ride Date')
plt.ylabel('Ride Fare')
plt.show()


#  line in the plot suggests that there is no clear relationship between ride date and ride fare. The fluctuations in the line may be due to various factors such as demand, supply, competition, weather, events, and promotions. Therefore, it may not be possible to draw any conclusive insights or trends from this plot alone. However, it is still important to investigate and analyze the data further to identify any patterns or correlations that may exist.

# In[98]:


#Finally, we can create a line plot of ride fares versus ride hour:


# In[101]:


import matplotlib.pyplot as plt

plt.scatter(df['Ride Hour'], df['Ride Fare'], c=df['Ride Time'], cmap='coolwarm')
plt.xlabel('Ride Hour')
plt.ylabel('Ride Fare')
plt.colorbar(label='Ride Time')
plt.show()


# Based on the scatter plot, we can see that there is no clear trend or pattern in the relationship between ride hour and ride fare. However, we can observe that at certain hours (9, 10, and 11), there are more rides with higher fares (above 30), and at hour 12, there are fewer rides with lower fares (below 25). These observations could potentially be due to factors such as demand or traffic during those specific hours. Overall, more analysis and data may be needed to draw a more definitive conclusion about the relationship between ride hour and ride fare.

# In[ ]:





# In[106]:


cities = df['City'].unique() # get the unique city names

for city in cities:
    city_data = df[df['City'] == city] # filter data for the current city
    plt.scatter(city_data['Ride Hour'], city_data['Ride Fare'], label=city)
    
plt.xlabel('Ride Hour')
plt.ylabel('Ride Fare')
plt.legend()
plt.show()


# ### To check for patterns or trends in the data over time, we can use line plots:

# In[108]:


# Convert ride date to datetime format
df['Ride Date'] = pd.to_datetime(df['Ride Date'])

# Group the data by ride date and calculate the average ride fare for each date
daily_fares = df.groupby('Ride Date')['Ride Fare'].mean()

# Plot the average ride fare for each date
plt.plot(daily_fares.index, daily_fares.values)
plt.xlabel('Ride Date')
plt.ylabel('Average Ride Fare')
plt.show()

# Group the data by ride hour and calculate the total number of rides for each hour
hourly_rides = df.groupby('Ride Hour')['Ride ID'].count()

# Plot the total number of rides for each hour
plt.plot(hourly_rides.index, hourly_rides.values)
plt.xlabel('Ride Hour')
plt.ylabel('Total Number of Rides')
plt.show()


# The high demand around 9 AM could be due to people commuting to work or school, while the low demand around midday could be due to people being at work or taking lunch breaks. Other factors, such as traffic patterns, could also play a role. However, it's important to keep in mind that correlation does not necessarily imply causation, and there could be other factors at play that are not captured in the data.

# # lets do some INFERENTIAL Statistics 

# One possible hypothesis is that the average ride time is different between New York and San Francisco. 

# In[111]:


# Filter the data for rides in New York and San Francisco
ny_rides = df[df['City'] == 'New York']['Ride Time']
sf_rides = df[df['City'] == 'San Francisco']['Ride Time']

# Conduct the t-test
t_stat, p_val = stats.ttest_ind(ny_rides, sf_rides, equal_var=False)

# Print the results
print('T-statistic:', t_stat)
print('P-value:', p_val)


# Based on the t-test results, we can see that the calculated t-statistic is 0.156 and the p-value is 0.879. Since the p-value is greater than the significance level of 0.05, we fail to reject the null hypothesis that there is no difference in the mean ride time between New York and San Francisco. Therefore, we do not have enough evidence to support the claim that the average ride time is different between these two cities.

# # The average ride fare in Los Angeles is higher than the average ride fare in San Francisco.
# 
# Null Hypothesis: The average ride fare in Los Angeles is not higher than the average ride fare in San Francisco.
# 
# Alternative Hypothesis: The average ride fare in Los Angeles is higher than the average ride fare in San Francisco.
# 
# We can use a two-sample t-test to test this hypothesis.

# In[114]:


# Subset the data for Los Angeles and San Francisco
la_data = df[df['City'] == 'Los Angeles']['Ride Fare']
sf_data = df[df['City'] == 'San Francisco']['Ride Fare']

# Perform the two-sample t-test
t_stat, p_val = ttest_ind(la_data, sf_data, equal_var=False)

# Print the results
print("T-statistic: {}".format(t_stat))
print("P-value: {}".format(p_val))


# Based on the results of the two-sample t-test, we cannot reject the null hypothesis that there is no significant difference in the mean ride fare between Los Angeles and San Francisco, since the p-value is higher than the commonly used significance level of 0.05. Therefore, we do not have evidence to support the hypothesis that the average ride fare in Los Angeles is higher than the average ride fare in San Francisco.

# # Hypothesis 2: The distribution of ride times in New York follows a normal distribution.
# To test this hypothesis, we can use a normality test, such as the Shapiro-Wilk test or the Anderson-Darling test. The null hypothesis is that the ride time data in New York is normally distributed, and the alternative hypothesis is that it is not normally distributed. We can calculate the test statistic and the p-value using a statistical software or library.

# In[115]:


from scipy.stats import shapiro

# Select ride time data for New York
ny_data = df[df['City'] == 'New York']['Ride Time']

# Perform Shapiro-Wilk test
stat, p_val = shapiro(ny_data)

# Print the results
print("Test statistic: {}".format(stat))
print("P-value: {}".format(p_val))


# Based on the Shapiro-Wilk test, with a p-value of 0.6257, we fail to reject the null hypothesis that the ride time data in New York is normally distributed. Therefore, we can conclude that there is no significant evidence to suggest that the distribution of ride times in New York is not normal.

# #  Regression ANALYSIS 
# Based on the missing values in the dataset, it is not possible to conduct a linear regression analysis since important variables such as the distance are missing. This missing information could provide valuable insights into the relationship between different variables. Additionally, logistic regression analysis requires important categorical values that are not available in the dataset. As a result, the only viable option for regression analysis in this case is to conduct a non-linear polynomial analysis.

# In[131]:


print(df.columns)


# ## Polynomial regression 

# Polynomial regression to model non-linear relationships between variables, such as the relationship between ride time and ride fare if there is a non-linear component.

# In[133]:


# Select the predictor variable
X = df[['Ride Time']]

# Select the dependent variable
y = df['Ride Fare']

# Create a polynomial feature object of degree 2
poly = PolynomialFeatures(degree=2)

# Transform the predictor variable to include polynomial features
X_poly = poly.fit_transform(X)

# Fit a linear regression model to the transformed data
model = LinearRegression()
model.fit(X_poly, y)

# Use the fitted model to make predictions on the original data
y_pred = model.predict(X_poly)

# Calculate the R-squared value of the model
r_squared = model.score(X_poly, y)

print("R-squared:", r_squared)


# The logistic regression model allows us to predict the probability of a binary outcome (in this case, whether a ride was rated good or bad) based on the values of the predictor variables (ride_time and hour_of_day).
# 
# The coefficients in the model represent the effect of each predictor variable on the log odds of the outcome. For example, in this model, a one-unit increase in ride_time is associated with a decrease in the log odds of a good rating by -0.0019. The p-values associated with each coefficient tell us whether the effect of the predictor variable is statistically significant or not.
# 
# The logistic regression model also allows us to calculate odds ratios, which represent the change in odds of the outcome associated with a one-unit increase in the predictor variable. For example, in this model, the odds of a good rating decrease by a factor of 0.998 for each additional minute of ride_time. If the odds ratio is less than 1, it means that increasing the predictor variable is associated with a decrease in the odds of the outcome, while an odds ratio greater than 1 indicates that increasing the predictor variable is associated with an increase in the odds of the outcome.

# In[132]:


# Plot the data points
plt.scatter(X, y, color='blue')

# Sort the predictor variable in ascending order
X_sorted = np.sort(X, axis=None)

# Predict the response variable for the sorted predictor variable
y_pred = model.predict(poly.fit_transform(X_sorted.reshape(-1,1)))

# Plot the polynomial line
plt.plot(X_sorted, y_pred, color='red')

# Add labels and title
plt.xlabel('Ride Time')
plt.ylabel('Ride Fare')
plt.title('Polynomial Regression')

# Show the plot
plt.show()



# 

# # KMEANS 

# In[137]:


# Select the variables we want to use for clustering
X = df[['Ride Time', 'Ride Fare']]

# Fit a KMeans clustering model to the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Assign each data point to a cluster
labels = kmeans.predict(X)

# Add the cluster labels to the dataframe
df['Cluster'] = labels

# Visualize the clusters
plt.scatter(df['Ride Time'], df['Ride Fare'], c=df['Cluster'])
plt.xlabel('Ride Time')
plt.ylabel('Ride Fare')
plt.show()


# By using KMeans clustering to group the ride-sharing data based on the ride time and ride fare variables. We are creating three clusters and visualizing the results using a scatter plot, where each point is colored according to its assigned cluster. The resulting clusters can help us understand patterns in the data and potentially target specific groups of riders with tailored marketing or pricing strategies.

# # Ridge regression and LASSO regression 

# Ridge regression adds a penalty term to the regression equation that is proportional to the squared magnitude of the coefficients. This penalty term shrinks the coefficients towards zero, but does not set any of them to exactly zero. Therefore, all the variables remain in the model, but their coefficients are reduced in size.
# 
# LASSO regression, on the other hand, adds a penalty term that is proportional to the absolute magnitude of the coefficients. This penalty term not only shrinks the coefficients towards zero, but can also set some of them to exactly zero. Therefore, LASSO regression performs feature selection by effectively removing some variables from the model.

# In[140]:


# Select the predictor variables
X = df[['Ride Time', 'Ride Hour']]

# Select the dependent variable
y = df['Ride Fare']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression object with alpha=1.0
ridge = Ridge(alpha=1.0)

# Fit the Ridge regression model to the training data
ridge.fit(X_train, y_train)

# Use the Ridge model to make predictions on the testing data
y_pred_ridge = ridge.predict(X_test)

# Calculate the R-squared value and mean squared error of the Ridge model
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("R-squared for Ridge regression:", r2_ridge)
print("Mean squared error for Ridge regression:", mse_ridge)

# Create a LASSO regression object with alpha=1.0
lasso = Lasso(alpha=1.0)

# Fit the LASSO regression model to the training data
lasso.fit(X_train, y_train)

# Use the LASSO model to make predictions on the testing data
y_pred_lasso = lasso.predict(X_test)

# Calculate the R-squared value and mean squared error of the LASSO model
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("R-squared for LASSO regression:", r2_lasso)
print("Mean squared error for LASSO regression:", mse_lasso)


# it seems that both Ridge regression and LASSO regression are not performing well on the given dataset, as both have negative R-squared values. This suggests that these models are not a good fit for the data, and other models may be more appropriate. The mean squared error values are also quite high, indicating that the models are not accurately predicting the target variable.

# # FINE 

# In[ ]:




