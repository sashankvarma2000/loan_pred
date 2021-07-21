

# Code to read csv file into colaboratory:


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# %matplotlib inline

plt.style.use('bmh')

data=pd.read_csv("accepted_2007_to_2018Q4.csv")

data

data.head()

data.tail()

data.info()

data.describe()

data.isnull().any()

data.isnull().sum()

data.isnull().sum()

#@title Default title text
data2 = data[[column for column in data if data[column].count() / len(data) >= 0.8]]


for c in data.columns:
    if c not in data2.columns:
        print(c, end=", ")
print('\n')
data = data2
data.shape

data

data.isnull().sum()

data.hist(figsize=(32, 40), bins=50, xlabelsize=8, ylabelsize=8);

data.shape

data

drop_attributes = ['id', 'issue_d', 'pymnt_plan','purpose','title', 'url','addr_state','earliest_cr_line', 'revol_util','initial_list_status', 'hardship_flag', 'disbursement_method','debt_settlement_flag','emp_title','term','emp_length','home_ownership','verification_status','loan_status','zip_code','last_pymnt_d','last_credit_pull_d','application_type']
len(drop_attributes)

data.drop(labels=drop_attributes, axis=1, inplace=True)

data

data.isnull().sum()

data.fillna(data.mean(), inplace=True)

data.isnull().sum()

data.dtypes



data

data.dropna(inplace=True)

grade = {'A': 1, 'B': 2, 'C': 3,'D':4,'E':5,'F':6,'G':7}
data.grade = [grade[item] for item in data.grade]
print(data)

data=data.drop(['sub_grade'],axis=1)

y=data.iloc[:,5:6]
y

x=data.loc[:,data.columns!='grade'].values
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
Rf=RandomForestRegressor(n_estimators=10,random_state=0,n_jobs=-1)
Rf.fit(x_train,y_train)

y_pred=Rf.predict(x_test)
y_pred

y_test

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

with open('loan_pred_model.pkl','wb') as f:
  pickle.dump(Rf,f)

with open('loan_pred_model.pkl','rb') as f:
  mp=pickle.load(f)

