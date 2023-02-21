#!/usr/bin/env python
# coding: utf-8

# ### Dependencies

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import Lasso # , LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Export dependencies list and overwrite requirements.txt with --force
get_ipython().system(' jupyter nbconvert --output-dir="./requirements" --to script C:\\datascience\\ml_supervised_linearregression_01\\RegressionModel_LinearRegression_Example.ipynb')
get_ipython().system(' cd requirements')
get_ipython().system(' pipreqs --force')


# ## <span style='background: lightblue'> 1. Data Collection </span>
# Load sample dataset. Shape: 731 rows x 4 columns

# In[2]:


# Initial view of the dataset shows that temperature, humidity, and windspeed could be used as predictors for the target variable rentals.
bikes_df = pd.read_csv("./data/bikes.csv")
bikes_df


# ## <span style='background: lightblue'> 2. Data Exploration </span>

# In[3]:


# None of the variables have a string type. No need to convert variable(s) to number to build the linear regression model.
bikes_df.info()


# In[4]:


bikes_df.describe()


# In[5]:


# Check for outliers
bikes_df['temperature'].plot.box(figsize = (5, 5))


# In[6]:


# Check for outliers
bikes_df['humidity'].plot.box(figsize = (5, 5))


# In[7]:


# Check for outliers
bikes_df['windspeed'].plot.box(figsize = (5, 5))


# #### Outlier results: For all 3 independent variables only 1 record for windspeed appears to be an outlier.

# ### <span style='background: lightgrey'>Linear Regression Assumption 1 of 5: Lack of Multicollinearity</span>
# The predictors should not be correlated to one another. The smaller the coefficient the smaller the correlation between 2 variables. 1 means a perfect positive correlation, 0 equals no correlation, and -1 means a perfect negative correlation.

# In[8]:


# For linear regression to work the features (X predictors) should be independent of each other. 

# To test correlation
bikes_df.corr()


# In[9]:


# Visualize the lack of multicollinearity
corr = bikes_df[['temperature', 'humidity', 'windspeed', 'rentals']].corr()
print('Pearson correlation coefficient matrix of each variable:\n', corr)

# Generate a mask for the diagonal cell
mask = np.zeros_like(corr, dtype=bool)
np.fill_diagonal(mask, val=True)

# Initialize matplotlib figure
fig, ax = plt.subplots(figsize=(5, 4))

# Generate figure
cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
cmap.set_bad('grey')

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
fig.suptitle('Pearson correlation coefficient matrix', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=10)
# fig.tight_layout()

print("A significant positive association of 0.8998 between the independent (X) variables only exists between windspeed and humidity. The rest of the associations between the independent variables are very weak. Later we'll see what effect this may have in the accuracy of the model.")


# In[10]:


# The scatter plot below confirms the high positive association between windspeed and humidity as shown in the Pearson correlation coefficient results.
bikes_df.plot(kind="scatter", x="windspeed", y="humidity")


# ### <span style='background: lightgrey'>Linear Regression Assumption 2(a) of 5: Linearity (before running regression)</span>
# Check if there is a linear relationship between y (dependent) and each X (independent) variable. A linear relationship is required for linear relationship models.

# In[11]:


# LINEARITY check

bikes_df.plot(kind="scatter", x="temperature", y="rentals")


# In[12]:


# LINEARITY check
bikes_df.plot(kind="scatter", x="humidity", y="rentals")


# In[13]:


# LINEARITY check
bikes_df.plot(kind="scatter", x="windspeed", y="rentals")


# In[14]:


#### Linearity results: All 3 independent variables appear to have a linear tendency.


# ## <span style='background: lightblue'>3. Data Preparation</span>
# Split dataset into y (output, target) and X (input, features) variables

# In[15]:


# y dependent (target) variable
output_var = "rentals"
y = bikes_df[[output_var]]
y


# In[16]:


# X independent (feature) variable
input_vars = list(bikes_df.columns) # get all column names
input_vars.remove(output_var) # remove target (y) column
X = bikes_df[input_vars] # assign new data frame to X
X


# In[17]:


# Run K-fold cross-validation on the full dataset
# It's essentially a random split
# On this case the dataset is split into 5 groups (fold) specified by n_splits = 5
# Each split contains 1 test data fold and 4 training data folds. The model is fit on the 4 training folds and the prediction is based on the remaining test fold. In each of the 5 iterations the test group (fold) is a different one.

X = bikes_df[input_vars]
y = bikes_df[[output_var]]

kf = KFold(n_splits=5, shuffle=True, random_state = 42) # random_state argument makes the results repeatable; # random_state = 42 will be removed in production
model = Lasso(alpha=0.1) # choose the alpha hyperparameter based on the model evaluation results in Step #5 to increase accuracy
cv_results = cross_val_score(model, X, y, cv=kf) # returns array of cross-validation scores
print(
    "r2 array: ", cv_results, '\n'
    "Mean of r2: ", np.mean(cv_results), '\n'
    "Std dev: ", np.std(cv_results), '\n'
    "95% confidence interval: ", np.quantile(cv_results, [0.025, 0.975]), '\n \n' 
    "The r2 score of 0.9826 means that 98.26% of our target (y-dependent variable) results can be explained by our 3 features (X-independent variables). This result along with a standard deviation of 0.0023 and a 95% confidence interval between 0.9798 and 0.9854 depict a highly accurate model."
)


# In[18]:


# Split data between training and test sets

# 25% of dataset is allocated to test set by default (test_size = 0.25)
# random_state argument makes the results repeatable
# random_state = 1234 argument will be removed in production
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 1234)


# In[19]:


# Check the resulting dimensions of all 4 tables
print("Training set (X, y): ", X_train.shape, y_train.shape) 
print("Test set (X, y): ", X_test.shape, y_test.shape)


# ## <span style='background: lightblue'>4. Train the Model (w/ train dataset)</span>

# In[20]:


model.fit(X_train, y_train) # LinearRegression() has an optional argument to normalize the data

print("The model was fitted.")


# In[21]:


# Estimate of the y-intercept

# y-intercept
model.intercept_


# In[22]:


# Estimate of the slope

# slope (regression coefficient)
# The slope results are listed in the order that they appear in the training data data frame. e.g., temperature, humidity, and windspeed
model.coef_


# ###  Below is the estimated regression line that best fits the data. With this information we can estimate what our model will predict in any weather condition (X inputs). This is the equation that can be used in production if our evaluation of the model proves to be satisfactory.
# 
# #### y = m * X + b
# 
# #### y = m * X - m * X - m * X + b
# #### y = (80.35 * X) - (-4665.74 * X) - (-196.22 * X) + 3800.68
# 
# #### e.g., If temp=72F, humidity=22%, windspeed=5mph
# #### y = (80.35 * 72) - (-4665.74 * .22) - (-196.22 * 5) + 3800.68
# #### = 7,578 rentals

# ## <span style='background: lightblue'>5. Evaluate the Model (test dataset)</span>

# In[23]:


# Calculate r2

# Score the predictions r2. The closer this value is to 1 the better the model is.
# r2 = r2_score(y_test, y_predicted) 
# r2 is alo called the coefficient of determination (measures accuracy)
r2 = model.score(X_test, y_test)

print(
    "r2: ", r2, "\n"
    "The r2 score of 0.9821 means that 98.21% of our target (y-dependent variable) results can be explained by our 3 features (X-independent variables) in the test dataset."
) 
# The r2 value of 0.9821 means that our model is able to explain 98.21% of the variability of the output of the test data.
# It tests the ability of the model to generalize. Again, the features explain about 98.21% of the target (rentals) variance.
# Note: The r2 has some pifalls since it's the result of our test set. We need to know what's the performance of the model to generalize with unseen data (e.g., training set). This is why in a previous step we calculated r2 with K-fold cross-validation on the full dataset.
# The mean of r2 using K-fold cross-validation on the full dataset equaled 98.26%
# r2 of the test set equaled 98.21%
# Both r2 results were very similar which support the accuracy of the model.


# In[24]:


# Compare predicted values versus the actual values

# Use our model to make predictions
y_predicted = model.predict(X_test)

# MAE (mean absolute error) 
mean_absolute_error(y_test, y_predicted) # The predictions of the model should be off the mark by an average of +/- the result


# In[25]:


# Concatenate the X and y variables in a data frame for visualization purposes
pd.concat([X_test, y_test], axis=1).reset_index()


# ### <span style='background: lightgrey'>Linear Regression Assumption 2(b) of 5: Linearity (after running regression)</span>
# Check if there is a linear relationship between y_test and y_prediction

# In[26]:


# LINEARITY check prep for next step
# Create dataframe with y_test and y_predicted

# y_test

y_predicted_df = pd.DataFrame(y_predicted, columns=["rentals"])
y_predicted_df


# In[27]:


# Append y_predicted column to original test set to review results

results = pd.concat([X_test, y_test], axis=1).reset_index()

results['y_predicted'] = y_predicted
results_df = results.rename(columns={'rentals':'y_test'})

results_df


# In[28]:


# LINEARITY check

X = y_test
y = y_predicted_df

plt.scatter(X, y)

# Plotting the diagonal line
p1 = max(max(X['rentals']), max(y['rentals'])) # - some interger will change x,y axis range
p2 = min(min(X['rentals']), min(y['rentals'])) # - some interger will change x,y axis range

plt.plot([p1, p2], [p1, p2],  # X and y points
         color='darkorange', linestyle='--')

plt.xlabel('y_test rentals')
plt.ylabel('y_predicted rentals')

plt.title('y_predicted vs. y_test')

print("There are appears to be a linear relationship between y_predicted and y_test. This reflects a highly accurate model.")


# ### <span style='background: lightgrey'>Linear Regression Assumption 3 of 5: Multivariate Normality</span>
# Check the normality of error distribution

# In[29]:


# Performing the test on the residuals

print(y_test.shape)
print(y_predicted.shape)

residuals = y_test - y_predicted.reshape(183, 1) # .reshape used to shape from (183, ) to (183, 1)
# Rename residual column
residuals = residuals.rename(columns={'rentals':'residual'})

p_value = normal_ad(residuals)[1] # TODO: Check what [1] is doing??????
print('p-value from the test Anderson-Darling test below 0.05 generally means non-normal:', p_value)

# Plot the residuals distribution
plt.subplots(figsize=(8, 4))
plt.title('Distribution of Residuals', fontsize=18)
sns.histplot(residuals, kde=True, stat='density')
plt.show()

# Report the normality of the residuals
if p_value < 0.05:
    print('The residuals are not normally distributed')
else:
    print('The residuals are normally distributed.')
    
print('Normally distributed residuals support the use of a linear regression model.')


# ### <span style='background: lightgrey'>Linear Regression Assumption 4 of 5: Homoscedasticity</span>
# Uniform variance for the residuals (prediction errors) of all data points

# In[30]:


# Review residuals and make sure the results make sense
residuals.reset_index(drop=True, inplace=True) # Reset residuals index to append column correctly and avoid NaN's
results_df['residuals'] = residuals
results_df


# In[31]:


# Plotting the residuals
plt.subplots(figsize=(9, 5))
plt.scatter(x=residuals.index, y=residuals.residual, alpha=0.8) # x = index number, y = residual value (residuals = y_test - y_predicted)

plt.plot(np.repeat(0, len(residuals.index)+2), color='darkorange', linestyle='--')

plt.ylabel('Residual', fontsize=14)
plt.xlabel('Rental Day Record', fontsize=14)
plt.title('Homoscedasticity Assumption', fontsize=16)
plt.grid(linewidth=0.3)
# plt.xlim([0, 183]) # specify x-axis range
sns.despine(left=True, bottom=True) # remove plot border
plt.show()  

print("The purpose of this graph is to show where residuals tend to concentrate. Residuals (prediction errors) should concentrate around the X-axis and be uniform.")


# In[32]:


results_df.describe()


# ### <span style='background: lightgrey'>Linear Regression Assumption 5 of 5: Independence of Observations<span>
# Check for no autocorrelation of residuals (errors) over time

# In[33]:


durbinWatson = durbin_watson(residuals)

print('Durbin-Watson:', durbinWatson)
if durbinWatson < 1.5:
    print('Signs of positive autocorrelation', '\n')
    print('Assumption not satisfied')
elif durbinWatson > 2.5:
    print('Signs of negative autocorrelation', '\n')
    print('Assumption not satisfied')
else:
    print('Little to no autocorrelation', '\n')
    print('Assumption satisfied')


# In[34]:


# Lasso regression

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()), # StandardScaler makes the mean of the distribution 0. About 68% of the values will lie be between -1 and 1. Is used to resize the distribution of values ​​so that the mean of the observed values ​​is 0 and the standard deviation is 1.
    ('model', Lasso())
])

# IMPORTANT: There is no point in picking alpha = 0 ('model__alpha' = 0), that is simply the linear regression.
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)}, # test several values from 0.1 to 10 with 0.1 step. For each value, we calculate the average value of the mean squared error in a 5-folds cross-validation and select the value of α that minimizes such average performance metrics
                      cv = 5, scoring="neg_mean_squared_error",verbose=0
) # verbose=0, to hide response in next step


# In[35]:


search.fit(X_train,y_train)


# In[36]:


alpha = search.best_params_
print("The best value for alpha was: ", alpha) # Selected automatically in 'model__alpha':np.arange(0.1,10,0.1) in the previous step
                                               # This value can be used to modify the alpha hyperparameter in Step #3 (data preparation section)

# This value can be used to compare which alpha is performing better; The closer to 0 MAE is the better
# MSE is more sensitive to outliers than MAE
mse = search.best_score_
print("Negative Mean Squared Error (-MSE) was: ", mse)

coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print("Asbolute value of coefficients: ", importance)

#===============================
#===============================
# print(np.array(features))
X = bikes_df[input_vars]
y = bikes_df[[output_var]]
features = X.columns
print(np.array(features)[importance > 0])
print(np.array(features)[importance == 0]) # If [] blank all features are important (no features equals 0)
#===============================
#===============================
_ = plt.plot(range(len(features)), importance)
_ = plt.xticks(range(len(features)), features, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()
print("The most important predictor, which the largest absolute value of the coefficients, is temperature. No features of 0 importance where found.")


# ## Conclusion
# 
# Although the bicycle rental dataset used is small I was able to obtain a high degree of accuracy of over 98% in both the full and test sets. Additionally, all 5 assumptions to check the validy of the linear model were satisfied (as noted in steps #1 to #5). Further review of this model and code is required and I don't consider this the final product. But as a general practice of supervised machine learning I think it's a good start.

# ## Next Steps
# 
# Remove the random_state arguments, deploy model into production, and re-check results.
