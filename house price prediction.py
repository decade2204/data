# 1.data cleaning and processing

# import modules
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns = None #Display complete print data
pd.options.display.max_rows = None
plt.rcParams['axes.unicode_minus']=False    # Used to display the negative sign normally
plt.rcParams['figure.dpi'] = 150 # Modify image resolution
plt.rcParams.update({'font.size': 16}) #Modify the overall font size of the image
import numpy as np
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn import ensemble
import xgboost as xgb # 1.5.0
import lightgbm as lgb # 2.2.3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #Codes as numbers
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings
warnings.filterwarnings("ignore") # ingore warnings

# read data
data = pd.read_csv('housing.csv')
data.head()

# shape of data
data.shape

import matplotlib.pyplot as plt
# Draw histogram
data.hist(bins=50, figsize=(20,15))

#Numerical characterization
data.describe().T

# # Check the number of missing values per column, no missing values
data.isnull().sum()

# Visualization of missing data
import missingno as msno
plt.figure(dpi=150)
p=msno.bar(data,color='chocolate',figsize=(25,8),fontsize=13)
plt.title('Missing data situation',fontsize=20)
plt.savefig('Missing data situation.png',dpi=300,bbox_inches = 'tight')# Saved as HD images

# View data type distribution
data.dtypes

# Encoding string data
dtypes_list=data.dtypes.values
columns_list=data.columns
for i in range(len(columns_list)):
    if dtypes_list[i]=='object':#number
        lb=LabelEncoder()
        lb.fit(data[columns_list[i]])
        data[columns_list[i]]=lb.transform(data[columns_list[i]])
data.head()

# Separate features and labels
# x = data.drop(['y1','y2','y3'],axis=1)
# y = data['y1']
x = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']
x.head()
y.head()

#  Data segmentation
# 7 to 3 divide the training set, test set, set random seed random_state, to ensure that the experiment can be reproduced

x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y,
                                                    test_size=0.3,
                                                    random_state=2022)

# Training set. Test set feature shape
x_train.shape,x_test.shape


# Visualization of latitude and longitude

# Show negative sign
plt.rcParams['axes.unicode_minus'] = False

x.plot(kind="scatter", x="longitude", y="latitude")
plt.xlabel('Longitude')
plt.ylabel('latitude')


data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3,
             s=x["population"]/50, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('latitude')


corr_matrix = data.corr()
print(corr_matrix) 

print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12, 8))


# 2.multi-model comparison
# Define model evaluation metrics
def try_different_method(model):
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    y_predict1 = model.predict(x_train)
    print('Training set evaluation metrics:')
    print('MSE values are: ', '%.4f'%float(mean_squared_error(y_train,y_predict1)))
    print('MAE values are: ', '%.4f'%float(mean_absolute_error(y_train,y_predict1)))
    print('RMSE values are:','%.4f'%float(np.sqrt(metrics.mean_squared_error(y_train, y_predict1))))
    print('MAPE values are:','%.4f'%float(np.mean(abs((y_train- y_predict1)/y_train)))) 
    print('R-Square values are: ', '%.4f'%float(metrics.r2_score(y_train, y_predict1)))
    
    print('Test set evaluation metrics:')
    print('MSE values are: ', '%.4f'%float(mean_squared_error(y_test,y_predict)))
    print('MAE values are: ', '%.4f'%float(mean_absolute_error(y_test,y_predict)))
    print('RMSE values are:','%.4f'%float(np.sqrt(metrics.mean_squared_error(y_test, y_predict))))
    print('MAPE values are:','%.4f'%float(np.mean(abs((y_test- y_predict)/y_test)))) 
    print('R-Square values are: ', '%.4f'%float(metrics.r2_score(y_test, y_predict)))



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
model_lr1 = LogisticRegression()
model_lr = LinearRegression()
model_XGB = xgb.XGBRegressor(random_state=2022,
                            verbosity=0,
                             n_jobs=-1,
                             max_depth=4, 
                            learning_rate=0.1, 
                            n_estimators=100)
print('Linear regression model scores are as follows: ')
try_different_method(model_lr)
print('XGboost scores are as follows:')
try_different_method(model_XGB)
print('Logistic regression model scores are as follows:')
try_different_method(model_lr1)

# Visualization of true and predicted values of linear regression, test set
y_test = y_test.reset_index(drop = True)
y_predict1 = model_lr.predict(x_test)
plt.figure(figsize = (12,8))
plt.plot(y_predict1,color = 'b',label = 'predict',markersize=8)
plt.plot(y_test,color = 'r',label = 'true',markersize=8)
plt.xlabel('Test Sample',fontsize=30)
plt.ylabel('y1',fontsize=30)
plt.title('Linear',fontsize=30)
# Coordinate axis font size
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25,loc='upper right')
plt.savefig('LR.png',dpi=300,bbox_inches = 'tight')

y_test = y_test.reset_index(drop = True)
y_predict2 = model_XGB.predict(x_test)
plt.figure(figsize = (12,8))
plt.plot(y_predict2,color = 'b',label = 'predict',markersize=8)
plt.plot(y_test,color = 'r',label = 'true',markersize=8)
plt.xlabel('Test Sample',fontsize=30)
plt.ylabel('y1',fontsize=30)
plt.title('XGBoost',fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25,loc='upper right')
plt.savefig('XGBoost.png',dpi=300,bbox_inches = 'tight')

# Visualization of true and predicted values of logicstic regression, test set
y_test = y_test.reset_index(drop = True)
y_predict3 = model_lr1.predict(x_test)
plt.figure(figsize = (12,8))
plt.plot(y_predict3,color = 'b',label = 'predict',markersize=8)
plt.plot(y_test,color = 'r',label = 'true',markersize=8)
plt.xlabel('Test Sample',fontsize=30)
plt.ylabel('y1',fontsize=30)
plt.title('Logicstic',fontsize=30)
# Coordinate axis font size
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25,loc='upper right')
plt.savefig('LR1.png',dpi=300,bbox_inches = 'tight')


# 3. Bayesian-XGBoost
# XGB-Bayesian Tuning
from skopt import BayesSearchCV 
opt = BayesSearchCV(xgb.XGBRegressor(verbosity=0,
                              n_jobs=-1,
                                random_state=2022), 
                    { 
                    'max_depth': Integer(3, 5), 
                    'n_estimators': Integer(100, 150), 
                    'learning_rate': (0.1, 0.3)},
                    random_state=2022,
                    scoring='r2',
                    n_iter=20,
                    cv=3)
opt.fit(x_train, y_train)
print(f'best params: {opt.best_params_}')

# BSXGB test set scores
BSXGB = xgb.XGBRegressor(random_state=2022,
                            verbosity=0,
                             n_jobs=-1,
                             max_depth=5, 
                            learning_rate=0.2760184491191485, 
                            n_estimators=150)
print('The BSXGB model scores are as follows: ')
try_different_method(BSXGB)

# Visualization of BSXGB true and predicted values, test set
y_test = y_test.reset_index(drop = True)
y_predict4 = BSXGB.predict(x_test)
plt.figure(figsize = (12,8))
plt.plot(y_predict4,color = 'b',label = 'predict',markersize=8)
plt.plot(y_test,color = 'r',label = 'true',markersize=8)
plt.xlabel('Test Sample',fontsize=30)
plt.ylabel('y1',fontsize=30)
plt.title('BSXGB',fontsize=30)

# Coordinate axis font size
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25,loc='upper right')
plt.savefig('BSXGB.png',dpi=300,bbox_inches = 'tight')
y_pred = pd.DataFrame(y_predict4,columns=['y_pred'])
aa = pd.concat([y_pred,y_test],axis=1)
aa.to_csv('BSXGB.csv',index=False)


# 4.SHAP
# Optimal model-SHAP, right mouse button to manually save HD DPI=300 images
# BSXGB
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300 # Modify image resolution
import shap # 0.39.0
shap.initjs()
explainer = shap.Explainer(BSXGB) # Enter your model
shap_values = explainer(x_test)# Input test set characteristics
#[Micro] Single-sample feature impact map I: waterfall
shap.plots.waterfall(shap_values[1],max_display=50) # Test set 2nd sample
# Force plot,[micro] Single-sample feature impact plot II: force plot
shap.plots.force(shap_values[7],matplotlib=True,figsize=(18, 4))# Sample 8 of the test set
# [macro] Feature density scatter plot: beeswarm
shap.plots.beeswarm(shap_values,max_display=50)
# [Macro] Feature importance SHAP value
shap.plots.bar(shap_values,max_display=50)
# summary
shap.summary_plot(shap_values, x_test,plot_type="bar", show = False)