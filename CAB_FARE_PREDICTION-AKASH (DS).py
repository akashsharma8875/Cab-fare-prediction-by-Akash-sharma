#!/usr/bin/env python
# coding: utf-8

# # Project Objective:

# ## Problem Statement 

# You are a cab rental start-up company. You have successfully run the pilot project and now want to launch your cab service across the country. You have collected the historical data from your pilot project and now have a requirement to apply analytics for fare prediction. You need to design a system that predicts the fare amount for a cab ride in the city.

# Importation of useful libraries

# In[1]:


import pandas as pd # Importing pandas for performing EDA
import os #getting access to input files
import numpy as np  # Importing numpy for Linear Algebric operations
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
import matplotlib.pyplot as plt # Importing for Data Visualization
from pprint import pprint
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression #ML algorithm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# Getting the Current Working directory

# In[2]:


os.getcwd()


# Loading the Required Train and Test dataset and showing the first 10 rows in data sets of both training and test sets

# In[3]:


Train_Cab=pd.read_csv('TRAIN_CAB.csv')


# In[4]:


Train_Cab.head(10)


# In[5]:


Test_Cab=pd.read_csv('test.csv')


# In[6]:


Test_Cab.head(10)


# After looking at the first glimpse of data, it can be stated that the train_cab data and test_cab data contain same kind of information apart the train_cab data has clearly mentioned the fare amount. the data is prepared upon some units such as date, distance and passenger count.

# # 1. DATA UNDERSTANDING

# Checking the Shape of both train and test datasets

# In[7]:


Train_Cab.shape


# In[8]:


Test_Cab.shape


# Checking the Datatypes of both train and test datasets

# In[9]:


Train_Cab.dtypes


# In[10]:


Test_Cab.dtypes


# With the datatypes, we can observe that the data types of both train and test cab datasets in fare amount and datetime is object types so we need to change the datatype of both in further step to make the process easy and understanding.

# In[11]:


Train_Cab.describe()


# In[12]:


Test_Cab.describe()


# # 2.Missing value analysis and Data Cleaning

# NOTE-The effective data cleaning is a vital part of analytics process to prepare and validate the data. a short word can be used here which is garbage in-garbage out (GIGO) which means if we go for analysis process without cleaning, a GIGO happens. overall it is needed to create an effective foundation for analysis process.

# In[13]:


#converting the dtype
Train_Cab['fare_amount']= pd.to_numeric(Train_Cab['fare_amount'], errors='coerce')


# In[14]:


Train_Cab.dtypes


# In[15]:


#converting the pickup object in datetime
from datetime import datetime
import calendar
Train_Cab['pickup_datetime']=pd.to_datetime(Train_Cab['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')


# In[16]:


Train_Cab.dtypes


# In[17]:


#getting info of each attribute
Train_Cab.info()


# In[18]:


#dropping the NA values in Datetime column
Train_Cab.dropna(subset= ["pickup_datetime"])


# In[19]:


#seperation of pickup_datetime in required field such as year, month, day of the week, day,hour and minute
Train_Cab['year']=Train_Cab['pickup_datetime'].dt.year
Train_Cab['day']=Train_Cab['pickup_datetime'].dt.day
Train_Cab['dayofweek']=Train_Cab['pickup_datetime'].dt.dayofweek
Train_Cab['month']=Train_Cab['pickup_datetime'].dt.month
Train_Cab['hour']=Train_Cab['pickup_datetime'].dt.hour
Train_Cab['minute']=Train_Cab['pickup_datetime'].dt.minute


# In[20]:


#rechecking the conversion
Train_Cab.dtypes


# The objects dtypes has been succesfully changed in above script

# In[21]:


Test_Cab['pickup_datetime']=pd.to_datetime(Test_Cab['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')


# In[22]:


#applying same covnersion of test_cab
Test_Cab['year']=Test_Cab['pickup_datetime'].dt.year
Test_Cab['day']=Test_Cab['pickup_datetime'].dt.day
Test_Cab['dayofweek']=Test_Cab['pickup_datetime'].dt.dayofweek
Test_Cab['month']=Test_Cab['pickup_datetime'].dt.month
Test_Cab['hour']=Test_Cab['pickup_datetime'].dt.hour
Test_Cab['minute']=Test_Cab['pickup_datetime'].dt.minute


# In[23]:


Test_Cab.dtypes


# ### Considerations

# 1.Passenger count should not be exceeded to 6(Note: without observations of SUV).
# 2.The outlier in pickup_datetime column of value 43.

# 3.As per the data set, longitudes range varies from -180 to 180 and latitudes ranges varies from -90 to 90.
# 
# Checking the datetime variables
# 

# In[24]:


#removing the datetime missing values in the rows to removing the adverse impact business impact
Train_Cab = Train_Cab.drop(Train_Cab[Train_Cab['pickup_datetime'].isnull()].index, axis=0)
print(Train_Cab.shape)
print(Train_Cab['pickup_datetime'].isnull())


# Checking the Passenger count variables

# In[25]:


Train_Cab["passenger_count"].describe()


# The descritpions show that there are 5345 passengers which is impractical. to make it practical, reducing the passenger count between 1 to 6 (while considering the SUV) 

# In[26]:


Train_Cab=Train_Cab[Train_Cab['passenger_count']<=6]
Train_Cab=Train_Cab[Train_Cab['passenger_count']>=1]


# In[27]:


Train_Cab['passenger_count'].describe()


# In[28]:


#remvoing the values of passengers with the value count of 0
Train_Cab = Train_Cab.drop(Train_Cab[Train_Cab["passenger_count"] == 0 ].index, axis=0)


# In[29]:


Train_Cab['passenger_count'].sort_values(ascending=True)


# In[30]:


Train_Cab.shape


# Checking the Fare amount Variables

# In[31]:


Train_Cab['fare_amount'].sort_values(ascending=False)


# In[32]:


Counter(Train_Cab['fare_amount']<0)


# In[33]:


Train_Cab = Train_Cab.drop(Train_Cab[Train_Cab["fare_amount"]<0].index, axis=0)
Train_Cab.shape


# In[34]:


#No negative values in the Fare_amount varibales
Train_Cab['fare_amount'].min()


# In[35]:


#removal of rows where the fare_amount=0
Train_Cab = Train_Cab.drop(Train_Cab[Train_Cab["fare_amount"]<1].index, axis=0)
Train_Cab.shape


# In[36]:


#observations shows that there is a huge difference in 1st, 2nd and 3rd fare_amount so we will limit it by considering the huge amount as outliers
Train_Cab = Train_Cab.drop(Train_Cab[Train_Cab["fare_amount"]> 454 ].index, axis=0)
Train_Cab.shape


# In[37]:


#elimination of rows for which the values of fare amount is missing in the datasets
Train_Cab = Train_Cab.drop(Train_Cab[Train_Cab['fare_amount'].isnull()].index, axis=0)
print(Train_Cab.shape)


# In[38]:


print(Train_Cab['fare_amount'].isnull().sum())


# In[39]:


Train_Cab['fare_amount'].describe()


# In[40]:


#clearing the longitude and lattitude rows by dropping out the range mentioned below:
#longitude(-90 to +90)
#lattitude(-180 to-180)
Train_Cab[Train_Cab['pickup_latitude']<-90]
Train_Cab[Train_Cab['pickup_latitude']>90]


# In[41]:


Train_Cab[Train_Cab['pickup_longitude']<-180]
Train_Cab[Train_Cab['pickup_longitude']>180]


# In[42]:


Train_Cab[Train_Cab['dropoff_latitude']<-90]
Train_Cab[Train_Cab['dropoff_latitude']>90]


# In[43]:


Train_Cab[Train_Cab['dropoff_longitude']<-180]
Train_Cab[Train_Cab['dropoff_longitude']>180]


# In[44]:


#index 5686 has the lattitude less than -90 so after dropping it, we get
Train_Cab=Train_Cab.drop((Train_Cab[Train_Cab['pickup_latitude']<-90]).index, axis=0)
Train_Cab=Train_Cab.drop((Train_Cab[Train_Cab['pickup_latitude']>90]).index, axis=0)
Train_Cab.shape


# In[45]:


Train_Cab.isnull().sum()


# In[46]:


Test_Cab.isnull().sum()


# So far, the data has been cleaned by exploring the practical aspects and with using multiple approaches, here are some high level approaches that are used is mentioned below:
# 
# 1.Get rid of unwanted observations:eliminated the rows in which the fare amount is missing.
# 
# 2.Fix structured error:(a).changed the Datatypes,(b).dropping the fare>0
# 
# 3.standardized the data:(a).changed the values count,(b).No negative values,(c).clearing the longitude and lattitude for easy understanding.
# 
# The process was done in both the test and train datasets.

# ### The data has been cleaned succesfully so now here we can proceed for the further data set operations

# Calculating the distance by using latlong

# In[47]:


#outlier analysis
plt.figure(figsize=(10,7))
plt.boxplot(Train_Cab['fare_amount'])
plt.xlabel('fare amount')
plt.ylabel('Count')
plt.title('Fare amount Boxplot')
plt.show()


# In[48]:


#as described in the above code script, the values of lattitude and longitude are given
#so formulatng the distance with the help of latlong which is also kn own as Haversine formula which will create a new variable called distance


#importing useful functions set from math 
from math import cos, sin, asin, sqrt,radians


# In[49]:


#calculating the distance between any two point(Specified in the decimal degrees)
def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    lon1,lat1,lon2,lat2=map(radians,[lon1,lat1,lon2,lat2])
    dlon=lon2-lon1
    dlat=lat2-lat1
    
    #formula implications(Haversine formula)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    #conversion in km
    km = 6371* c
    return km


# In[50]:


Train_Cab['distance'] = Train_Cab[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[51]:


Test_Cab['distance'] = Test_Cab[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[52]:


Train_Cab.head()


# In[53]:


Test_Cab.head()


# In[54]:


Train_Cab.nunique()


# In[55]:


Test_Cab.nunique()


# In[56]:


#setting the fare in ascending order to find whether the outliers are present or not
Train_Cab['distance'].sort_values(ascending=False)


# The above script shows that some of the values are very high which means that there are more than 8000 kms have been travelled by some of the passengers so clearly stated that the dataset has some outlier and it is needed to be removed

# In[57]:


Counter(Train_Cab['distance']==0)


# In[58]:


Counter(Test_Cab['distance']==0)


# In[59]:


Counter(Train_Cab['fare_amount']==0)


# In[60]:


#remvoing the rows holding the zero value for distance
Train_Cab=Train_Cab.drop(Train_Cab[Train_Cab['distance']==0].index, axis=0)


# In[61]:


Train_Cab.shape


# In[62]:


#removing the shape distance values>129 kms
Train_Cab=Train_Cab.drop(Train_Cab[Train_Cab['distance']>130].index, axis=0)


# In[63]:


Train_Cab.shape


# In[64]:


Train_Cab.head()


# The pickup_time data has been splitted in required format like month, year, dayofweek etc so we can drop the pickup_date time variables from both the train and test data. the other variables such as drop longitude and lattitude can be dropped as pickup distance is considering in both the datasets 

# In[65]:


#dropping the longitude and lattitude from both the train-test data
Drop_Train_Cab = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'minute']
Train_Cab = Train_Cab.drop(Drop_Train_Cab, axis = 1)


# In[66]:


Drop_Test_Cab = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'minute']
Test_Cab = Test_Cab.drop(Drop_Test_Cab, axis = 1)


# In[67]:


Train_Cab.head()


# In[68]:


Test_Cab.head()


# In[69]:


Train_Cab.dtypes


# In[70]:


#changing the dtypes of train dataset in required type
Train_Cab['passenger_count'] = Train_Cab['passenger_count'].astype('int64')
Train_Cab['year'] = Train_Cab['year'].astype('int64')
Train_Cab['month'] = Train_Cab['month'].astype('int64')
Train_Cab['dayofweek'] = Train_Cab['dayofweek'].astype('int64')
Train_Cab['day'] = Train_Cab['day'].astype('int64')
Train_Cab['hour'] = Train_Cab['hour'].astype('int64')


# In[71]:


Train_Cab.dtypes


# In[72]:


Test_Cab.dtypes


# # 3.Visual Observations

# In[73]:


plt.figure(figsize=(10,5))
sns.barplot(x='passenger_count',y='fare_amount',data=Train_Cab).set_title("Fare Amount vs passenger count")


# Observation-It can be stated that 2 is the most common passneger count with the high level of fare amount. Here the fare amount is measured according the distance travelled. A Cab with the passenger count=2 has travelled the most. Other passenger counts are showing almost same kind of observations

# In[74]:


plt.figure(figsize=(10,7))
Train_Cab.groupby(Train_Cab['hour'])['hour'].count().plot(kind='barh')
plt.show()


# Observation-Lowest cabs at early morning (5:00am) and and highest cab at the between office rush hours(18:00pm and 19:00pm)

# In[75]:


#A time and fare relation visualiazation
plt.figure(figsize=(15,7))
plt.scatter(x=Train_Cab['hour'], y=Train_Cab['fare_amount'], s=10)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# With the observation, it can be seen that the cab fare at early morning at 7:00 am and late night at 23:00 pm are the costliest.

# In[76]:


plt.figure(figsize=(14,7))
sns.distplot(Train_Cab['fare_amount']).set_title("Visualisation of Trip Fare")
Train_Cab.loc[Train_Cab['fare_amount']<0].shape
Train_Cab["fare_amount"].describe()


# In[77]:


#fare and days visualization
plt.figure(figsize=(10,5))
plt.scatter(x=Train_Cab['dayofweek'], y=Train_Cab['fare_amount'],s=10)
plt.xlabel('Day of week')
plt.ylabel('Fare')
plt.show()


# Highest fare on sunday, monday and thursday. lowest fare at wednesday and saturday.
# the cab fare is low and high demand of cabs on sunday and monday shows the high fare prices. may be low demands on weekend

# In[78]:


#impact of the day on the numbers of cab rides
plt.figure(figsize=(10,7))
sns.countplot(x='day', data=Train_Cab)
plt.show()


# Observation-It seems that cab rides are affected at the starting and ending of the month

# In[79]:


#relationship between distance and fare
plt.figure(figsize=(10,7))
plt.scatter(x=Train_Cab['distance'], y=Train_Cab['fare_amount'], c="r")
plt.xlabel('distance')
plt.ylabel('fare')
plt.show()


# The cab fare is increasing with the distance which is an obvious statement

# # 4.Feature Scaling:

# In[80]:


#checking the normalization whether the training data is uniformly data is uniformly distributed or not
for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(Train_Cab[i],bins='auto', color='blue')
    plt.title("Distribution for variable"+i)
    plt.ylabel('Density of Variable Amount')
    plt.show()


# The above observation shows that the skewness of target variable(distance, fare_amount) is high which is needed to be reduced by performing log transform

# In[81]:


Train_Cab['fare_amount']=np.log1p(Train_Cab['fare_amount'])
Train_Cab['distance']=np.log1p(Train_Cab['distance'])


# In[82]:


#rechecking if the data is uniformly distributed now or not
for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(Train_Cab[i],bins='auto', color='blue')
    plt.title("Distribution for variable"+i)
    plt.ylabel('Density of Variable Amount')
    plt.show()


# The bell shaped curved shown in the above visaulize area which means that the continuous variable are now normally distributed so there is no need of using the feature scalling technique i.e normalization or standardization 

# In[83]:


#similarly checking for the test data if it is uniformly distributed or not
sns.distplot(Test_Cab['distance'],bins='auto',color='red')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# skewness of the distance is high which can be reduced by applying the log transform

# In[84]:


Test_Cab['distance']=np.log1p(Test_Cab['distance'])


# In[85]:


#rechecking the distribution again
sns.distplot(Test_Cab[i],bins='auto',color='red')
plt.title("Distribution for Variable ")
plt.ylabel("Density")
plt.show()


# The bel shaped curved is again showing that there is no need of feature scalling i.e, Normalization or Standardization as the continuous variable are normally distributed

# # 5.Model Selection

# In[86]:


Train_Cab.head()


# In[87]:


x=Train_Cab[['passenger_count','year','day','dayofweek','month','hour','distance']]
y=Train_Cab[['fare_amount']]


# In[88]:


#spliting the dataset for further process
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.20, random_state = 1)


# In[89]:


xtrain.shape


# In[90]:


xtest.shape


# In[91]:


ytrain.shape


# In[92]:


ytest.shape


# # Linear Regression Model:

# In[93]:


LR_Fit_Cab= LinearRegression().fit(xtrain,ytrain)


# In[94]:


#prediction on both train and test data
pred_Train_LR=LR_Fit_Cab.predict(xtrain)
pred_Test_LR=LR_Fit_Cab.predict(xtest)


# In[95]:


##calculating RMSE for train data
RMSE_Train_LR = np.sqrt(mean_squared_error(ytrain, pred_Train_LR))

##calculating RMSE for test data
RMSE_Test_LR = np.sqrt(mean_squared_error(ytest, pred_Test_LR))


# In[96]:


print("Root Mean Squared Error For Training data = "+str(RMSE_Train_LR))
print("Root Mean Squared Error For Test data = "+str(RMSE_Test_LR))


# In[97]:


## R^2 calculation for train data
r2_score(ytrain, pred_Train_LR)


# In[98]:


## R^2 calculation for train data
r2_score(ytest, pred_Test_LR)


# ## Gradient Boosting Model:

# In[99]:


#building a model on top of training dataset
GB_Fit_Cab=GradientBoostingRegressor().fit(xtrain,ytrain)


# In[100]:


#prediction on both train and test data
pred_Train_GB=GB_Fit_Cab.predict(xtrain)
pred_Test_GB=GB_Fit_Cab.predict(xtest)


# In[101]:


#calculating the RMSE(root mean squared error) for train and test data
RMSE_Train_GB=np.sqrt(mean_squared_error(ytrain, pred_Train_GB))
RMSE_Test_GB=np.sqrt(mean_squared_error(ytest, pred_Test_GB))
print("RMSE for training data="+str(RMSE_Train_GB))
print("RMSE for training data="+str(RMSE_Test_GB))


# In[102]:


#calculating the R2 score for train and test data
r2_score(ytrain, pred_Train_GB)


# In[103]:


r2_score(ytest,pred_Test_GB)


# # Decision Tree Model:

# In[104]:


DT_Fit_Cab= DecisionTreeRegressor(max_depth = 2).fit(xtrain,ytrain)


# In[105]:


#prediction on both train and test data
pred_Train_DT=DT_Fit_Cab.predict(xtrain)
pred_Test_DT=DT_Fit_Cab.predict(xtest)


# In[106]:


##calculating RMSE for train data
RMSE_Train_DT = np.sqrt(mean_squared_error(ytrain, pred_Train_DT))

##calculating RMSE for test data
RMSE_Test_DT = np.sqrt(mean_squared_error(ytest, pred_Test_DT))


# In[107]:


print("Root Mean Squared Error For Training data = "+str(RMSE_Train_DT))
print("Root Mean Squared Error For Test data = "+str(RMSE_Test_DT))


# In[108]:


## R^2 calculation for train data
r2_score(ytrain, pred_Train_DT)


# In[109]:


## R^2 calculation for train data
r2_score(ytest, pred_Test_DT)


# # Random Forest Model:

# In[110]:


RF_Fit_Cab= RandomForestRegressor(n_estimators=200).fit(xtrain,ytrain)


# In[111]:


#prediction on both train and test data
pred_Train_RF=RF_Fit_Cab.predict(xtrain)
pred_Test_RF=RF_Fit_Cab.predict(xtest)


# In[112]:


##calculating RMSE for train data
RMSE_Train_RF = np.sqrt(mean_squared_error(ytrain, pred_Train_RF))

##calculating RMSE for test data
RMSE_Test_RF = np.sqrt(mean_squared_error(ytest, pred_Test_RF))


# In[113]:


print("Root Mean Squared Error For Training data = "+str(RMSE_Train_RF))
print("Root Mean Squared Error For Test data = "+str(RMSE_Test_RF))


# In[114]:


## R^2 calculation for train data
r2_score(ytrain, pred_Train_RF)


# In[115]:


## R^2 calculation for train data
r2_score(ytest, pred_Test_RF)


# # New Results Optimizations With Parameters Tuning:

# In[116]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[117]:


##Random Hyperparameter Grid
from sklearn.model_selection import train_test_split,RandomizedSearchCV


# In[118]:


##Random Search CV on Random Forest Model
RRF = RandomForestRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))


# In[119]:


rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}

randomcv_rf = RandomizedSearchCV(RRF, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_rf = randomcv_rf.fit(xtrain,ytrain)
predictions_RRF = randomcv_rf.predict(xtest)
view_best_params_RRF = randomcv_rf.best_params_
best_model = randomcv_rf.best_estimator_
predictions_RRF = best_model.predict(xtest)


# In[120]:


#R^2
RRF_r2 = r2_score(ytest, predictions_RRF)
#Calculating RMSE
RRF_rmse = np.sqrt(mean_squared_error(ytest,predictions_RRF))
print('Random Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_RRF)
print('R-squared = {:0.2}.'.format(RRF_r2))
print('RMSE = ',RRF_rmse)


# In[121]:


#applying same approach with other used modules
GB = GradientBoostingRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(GB.get_params())


# In[122]:


##Random Search CV on gradient boosting model
GB = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(1,20,2))
depth = list(range(1,100,2))
# Create the random grid
rand_grid = {'n_estimators': n_estimator,
               'max_depth': depth}
randomcv_GB = RandomizedSearchCV(GB, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)
randomcv_GB = randomcv_GB.fit(xtrain,ytrain)
predictions_GB = randomcv_GB.predict(xtest)
view_best_params_GB = randomcv_GB.best_params_
best_model = randomcv_GB.best_estimator_
predictions_GB = best_model.predict(xtest)


# In[123]:


#R^2
GB_r2 = r2_score(ytest, predictions_GB)
#Calculating RMSE
GB_rmse = np.sqrt(mean_squared_error(ytest,predictions_GB))
print('Random Search CV Gradient Boosting Model Performance:')
print('Best Parameters = ',view_best_params_GB)
print('R-squared = {:0.2}.'.format(GB_r2))
print('RMSE = ', GB_rmse)


# In[124]:


from sklearn.model_selection import GridSearchCV    
## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(xtrain,ytrain)
view_best_params_GRF = gridcv_rf.best_params_

#Apply model on test data
predictions_GRF = gridcv_rf.predict(xtest)

#R^2
GRF_r2 = r2_score(ytest, predictions_GRF)
#Calculating RMSE
GRF_rmse = np.sqrt(mean_squared_error(ytest,predictions_GRF))

print('Grid Search CV Random Forest Regressor Model Performance:')
print('Best Parameters = ',view_best_params_GRF)
print('R-squared = {:0.2}.'.format(GRF_r2))
print('RMSE = ',(GRF_rmse))


# In[125]:


## Grid Search CV for gradinet boosting
GB = GradientBoostingRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))

# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}

## Grid Search Cross-Validation with 5 fold CV
gridcv_GB = GridSearchCV(GB, param_grid = grid_search, cv = 5)
gridcv_GB = gridcv_GB.fit(xtrain,ytrain)
view_best_params_GGB = gridcv_GB.best_params_

#Apply model on test data
predictions_GGB = gridcv_GB.predict(xtest)

#R^2
GGB_r2 = r2_score(ytest, predictions_GGB)
#Calculating RMSE
GGB_rmse = np.sqrt(mean_squared_error(ytest,predictions_GGB))

print('Grid Search CV Gradient Boosting regression Model Performance:')
print('Best Parameters = ',view_best_params_GGB)
print('R-squared = {:0.2}.'.format(GGB_r2))
print('RMSE = ',(GGB_rmse))


# # Predicting the Fare from Test_Cab Dataset:

# The Dataset(train and test) has been cleaned and tested by applying the sutiable algorithm and now, the prediction will be done with the help of using the Random Forest model as it shows overall good r2 score which is around 0.96.

# In[126]:


## Grid Search CV for random Forest model
regr = RandomForestRegressor(random_state = 0)
n_estimator = list(range(11,20,1))
depth = list(range(5,15,2))


# In[127]:


# Create the grid
grid_search = {'n_estimators': n_estimator,
               'max_depth': depth}


# In[128]:


## Grid Search Cross-Validation with 5 fold CV
gridcv_rf = GridSearchCV(regr, param_grid = grid_search, cv = 5)
gridcv_rf = gridcv_rf.fit(xtrain,ytrain)
view_best_params_GRF = gridcv_rf.best_params_


# In[129]:


predictions_GRF_test_Df = gridcv_rf.predict(Test_Cab)


# In[130]:


predictions_GRF_test_Df


# In[131]:


Test_Cab['Predicted_fare'] = predictions_GRF_test_Df


# In[132]:


Test_Cab.head(20)


# In[133]:


Test_Cab['Predicted_fare'].mean()


# In[134]:


Test_Cab['distance'].mean()


# Here, we have successfully predicted the fare of cab with an average fare ($2.35) for average 1.27 Km

# In[135]:


Test_Cab.to_csv('Test_Cab.csv')


# # Thank You
