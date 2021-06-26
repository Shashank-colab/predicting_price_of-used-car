#Linear Regression and Random Forest Case Study
# Predicting the price of pre owned car
#importing necessary libreries
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#setting dimensions for plot


#changing directory
os.chdir("C:/Users/SHASHANAK/Downloads")

#reading csv file
cars_data = pd.read_csv('cars_sampled.csv')

#creating a copy
cars = cars_data.copy()

#understanding the structure of data
cars.info()

#summarizing the data
cars.describe()
pd.set_option('display.float_format', lambda x:('% .3f')%x)
cars.describe()

#to display max columns
pd.set_option('display.max_columns', 500)

#deleting unwanted columns
col = ['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars = cars.drop(col,axis=1)

#removing duplicate records
cars.drop_duplicates(keep='first',inplace=True)

#no.of missing values in columns
cars.isnull().sum()

#variable yearofRegistration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot('yearOfRegistration','price',data=cars,scatter=True,fit_reg=False)
#working range of yearof registration is 1950 to 2018

#variable Powerps
power_count = cars['powerPS'].value_counts().sort_index()
cars['powerPS'].describe()
sns.distplot(cars['powerPS'])
sns.boxplot(y= cars['powerPS'])
sns.regplot(cars['powerPS'],cars['price'],fit_reg=False)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#working range of powerPS is 10 to 500

#variable Price
price_count = cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
sns.boxplot(y = cars['price'])
cars['price'].describe()
sum(cars['price']>150000)
sum(cars['price']<100)
#working range of price is 100 to 150000

#now we work on range of data
cars = cars[(cars.yearOfRegistration <=2018)
            &(cars.yearOfRegistration >= 1950)
            &(cars.price >=100)&(cars.price <= 150000)
            &(cars.powerPS >= 10)
            &(cars.powerPS <= 500)]

#for further simplication we merge year and month of registration
#creating new variable age
cars['age'] = (2018 - cars['yearOfRegistration']) + (cars['monthOfRegistration']/12)
cars['age'] = round(cars['age'],2)
cars['age'].describe()

#dropping the yearOfRegistration and monthOfregistartion
cars = cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualizing parameters
#age
sns.distplot(cars['age'],color='green')
sns.boxplot(y = cars['age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y = cars['price'])

#PowerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y= cars['powerPS'])

# age vs price
sns.regplot('age','price',data= cars,scatter = True,fit_reg=False)
#price are higher for newer cars
#with increase inage price of cars decrease
#however some cars are priced higher with increase in age

#price vs powerPS
sns.regplot('powerPS','price',data=cars,scatter=True,fit_reg=False)
#with increase in powerPS price of cars also increase

#variable kilometer
km_count = cars['kilometer'].value_counts().sort_index()
print(km_count)
pd.crosstab(cars['kilometer'],'count',normalize=True)
sns.boxplot(y = cars['kilometer'])
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot('kilometer','price',data=cars,scatter=True,fit_reg=False)
#consider in modelling

#variable seller
seller_count = cars['seller'].value_counts().sort_index()
print(seller_count)
#only one car is commercial rest are private so seller column is insignificant

#variable offerType
offer_type_count = cars['offerType'].value_counts().sort_index()
print(offer_type_count)
#all cars have offers #insinificant

# variable abtest
abtest_count = cars['abtest'].value_counts().sort_index()
print(abtest_count)
pd.crosstab(cars['abtest'], columns='count')
sns.countplot(cars['abtest'])
#equally distributed
sns.boxplot(x=cars['abtest'],y=cars['price'])
#for ever price value there is almost 50-50 distribution
#does not affect price #insignificant

#variable vehicleType
vehicle_type_count = cars['vehicleType'].value_counts().sort_index()
print(vehicle_type_count)
pd.crosstab(cars['vehicleType'],'count',normalize=True)
sns.countplot(cars['vehicleType'])
sns.boxplot('vehicleType','price',data=cars)
#8 types of cars are there #limosine having max frequency
#vehicleType affects price

#variable gearbox
gearbox_count = cars['gearbox'].value_counts().sort_index()
print(gearbox_count)
pd.crosstab(cars['gearbox'],'count',normalize=True)
sns.countplot(cars['gearbox'])
sns.boxplot('gearbox','price',data = cars)
#gearbox affects price

#variable model
model_count = cars['model'].value_counts().sort_index()
print(model_count)
pd.crosstab(cars['model'],'count',normalize=True)
sns.countplot(cars['model'])
sns.boxplot(x=cars['model'],y=cars['price'])
#cars are distributed over many models
#considered in modelling

#variable fuelType
fueltype_count = cars['fuelType'].value_counts().sort_index()
print(fueltype_count)
pd.crosstab(cars['fuelType'],'count',normalize=True)
sns.countplot(cars['fuelType'])
sns.boxplot('fuelType','price',data = cars)
#fueltype affect price

#variable brand
brand_count = cars['brand'].value_counts().sort_index()
print(brand_count)
pd.crosstab(cars['brand'],'count',normalize=True) 
sns.countplot(cars['brand'])
sns.boxplot(cars['brand'],cars['price'])
#cars are distributed over many brands
#considered for modelling

#variable notrepairdamage
#yes = car is damaged but not rectified
#no = car was damaged but rectified
notrepair_count = cars['notRepairedDamage'].value_counts().sort_index()
print(notrepair_count)
pd.crosstab(cars['notRepairedDamage'],'count',normalize=True)
sns.countplot(cars['notRepairedDamage'])
sns.boxplot('notRepairedDamage','price',data = cars)
#cars that require repair work are priced lower

#now removing insignificant variable
col = ['seller','offerType','abtest']
cars = cars.drop(col,axis=1)
cars_copy = cars.copy()

#now checking corelation
correlation = cars.corr()
cars.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

"""we are going to build a linear regression and random forest model on two 
sets of data .
1. data obtained by omitting rows with any missig values.
2.data obtained by inputting the missing values."""

#going with omitting the rows of missing values
cars_omit = cars.dropna(axis=0)

#converting categorical variable to dummy variable
cars_omit = pd.get_dummies(cars_omit,drop_first=True)

#building model with omitted data
#separating input and output data
x1 = cars_omit.drop('price',axis=1,inplace=False)
y1 = cars_omit['price']

#plotting the variable price
prices = pd.DataFrame({'1.before':y1,'2.after':np.log(y1)})
prices.hist()
#we do log of prices so that we have proper distribution

#now transforming prices as a logarithmic value
y1 = np.log(y1)

#splitting the dataset inta test and train
train_x1,test_x1,train_y1,test_y1 = train_test_split(x1,y1,test_size = 0.3,random_state=3)

#baseline model for omitted data
"""we are making the base model by using the test data mean value this is 
to set a benchmark and to compare with our regression model"""

#finding the mean for the test data value
base_pred = np.mean(test_y1)

#repeating the same value till the length of data
base_pred = np.repeat(base_pred,len(test_y1))

#finding the root mean squared error
base_rms_error = np.sqrt(mean_squared_error(test_y1,base_pred))
print(base_rms_error)

'''linear regression with omitted data'''
lgr = LinearRegression(fit_intercept=True)

#model
model_lin1 = lgr.fit(train_x1,train_y1)

#predicting model on test set
car_prediction_lin1 = lgr.predict(test_x1)

#computing the mean squared error and root mean squared error
lin_mse_1 = mean_squared_error(test_y1,car_prediction_lin1)
print(lin_mse_1)
lin_rmse_1 = np.sqrt(lin_mse_1)
print(lin_rmse_1)

#R squared value
r2_lin_test_1 = model_lin1.score(test_x1,test_y1)
r2_lin_train_1 = model_lin1.score(train_x1,train_y1)
print(r2_lin_test_1,r2_lin_train_1)

#regression diagnostics - residual plot analysis
residuals1 = test_y1 - car_prediction_lin1
sns.regplot(car_prediction_lin1,residuals1,scatter = True,fit_reg = False)
residuals1.describe()

'''random forest with omitted data'''
rf = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth =100,
                           random_state = 1,min_samples_leaf = 4,min_samples_split =10)
#model
model_rf_1 = rf.fit(train_x1,train_y1)
#predicting on test set
car_prediction_rf1 = rf.predict(test_x1)

#computing mse and rmse value
rf_mse_1 = mean_squared_error(test_y1,car_prediction_rf1)
rf_rmse_1 = np.sqrt(rf_mse_1)
print(rf_rmse_1)

#computing r squared value
r2_rf_test1 = model_rf_1.score(test_x1,test_y1)
r2_rf_train_1 = model_rf_1.score(train_x1,train_y1)
print(r2_rf_test1,r2_rf_train_1)

#random forest analysis residual plot analysis
rf_residual = test_y1 - car_prediction_rf1
sns.regplot(car_prediction_rf1,rf_residual)


'''linear regression with imputed data'''
cars_imputed = cars.apply(lambda x: x.fillna(x.median()) if x.dtype == 'float' 
                          else x.fillna(x.value_counts().index[0]))
cars.isnull().sum()
cars_imputed.isnull().sum()

#converting categorical variable to dummy variable
cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

#separating imput and out put variable
x2 = cars_imputed.drop(['price'],axis = 'columns')
y2 = cars_imputed['price']

# plotting price variable
prices2 = pd.DataFrame({'1.before':y2,'2.after':np.log(y2)})
prices2.hist()

#changing price as log value
y2 = np.log(y2)

#splitting the data in test and train
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3,random_state=3)

#baseline model for imputed data
#we are creating a baseline model by using the test data mean value this
#is done to set a benchmark and to compare with our regression model
#finding the mean of test value
base_pred2 = np.mean(test_y2) 
base_pred2 = np.repeat(base_pred2, len(test_y2))

#finding the rmse
base_rms_error_imputed = np.sqrt(mean_squared_error(test_y2,base_pred2))

#now linear regression with imputed data
lgr2 = LinearRegression()

#fitting training data
model_lin2 = lgr2.fit(train_x2,train_y2)

#predicting model on test set
car_prediction_lin2 = lgr2.predict(test_x2)

#calculating mse and rmse value
lin_rmse_2 = np.sqrt(mean_squared_error(test_y2, car_prediction_lin2))
print(lin_rmse_2)

# R squared value
r2_lin_test_2 = model_lin2.score(test_x2,test_y2)
r2_lin_train_2 = model_lin2.score(train_x2,train_y2)
print(r2_lin_test_2,r2_lin_train_2)

# regression diagnostic _residual plot analysis
residual2 = test_y2 - car_prediction_lin2
sns.regplot(car_prediction_lin2,residual2,scatter = True,fit_reg=False)
residual2.describe()

'''random forest with imputed data'''
#model parameters
rf2 = RandomForestRegressor(min_samples_split=10,n_estimators=100,
                            min_samples_leaf=4,random_state=1,max_depth=100)
 
#model
model_rf_2 = rf2.fit(train_x2,train_y2)

#predicting model on test set
car_prediction_rf2 = rf2.predict(test_x2)

#computing mse and rmse value
rf2_mse_2 = mean_squared_error(test_y2,car_prediction_rf2)
rf2_rmse_2 = np.sqrt(rf2_mse_2)
print(rf2_mse_2,rf2_rmse_2)

#R squared value
r2_rf_test_2 = model_rf_2.score(test_x2,test_y2)
r2_rf_train_2 = model_rf_2.score(train_x2,train_y2)
print(r2_rf_test_2,r2_rf_train_2)

#random forest diagnostics - residual plot analysis
rf_residual_2 = test_y2 - car_prediction_rf2
sns.regplot(car_prediction_rf2,rf_residual_2)
rf_residual_2.describe()


##############CONCLUSION#######################
''' In conclusion for we can use RANDOM forest model with dropping the null values'''