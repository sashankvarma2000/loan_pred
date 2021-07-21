import pandas as pd
import category_encoders
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def loan_prediction(loan_amt,annual_inc,int_rate,tot_cur_bal):
    data=pd.read_csv("accepted_2007_to_2018Q4.csv")
    data2 = data[[column for column in data if data[column].count() / len(data) >= 0.8]]


    for c in data.columns:
        if c not in data2.columns:
            print(c, end=", ")
    print('\n')
    data = data2
    drop_attributes = ['id', 'issue_d', 'pymnt_plan','purpose','title', 'url','addr_state','earliest_cr_line', 'revol_util','initial_list_status', 'hardship_flag', 'disbursement_method','debt_settlement_flag','emp_title','term','emp_length','home_ownership','verification_status','loan_status','zip_code','last_pymnt_d','last_credit_pull_d','application_type']
    data.drop(labels=drop_attributes, axis=1, inplace=True)
    data.fillna(data.mean(), inplace=True)
    data.dropna(inplace=True)
    grade = {'A': 1, 'B': 2, 'C': 3,'D':4,'E':5,'F':6,'G':7}
    data.grade = [grade[item] for item in data.grade]
    data=data.drop(['sub_grade'],axis=1)
    y=data.iloc[:,5:6]
    x=data.loc[:,data.columns!='grade'].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    Rf=RandomForestRegressor(n_estimators=10,random_state=0,n_jobs=-1)
    Rf.fit(x_train,y_train)
    return Rf.predict(x_test)
loan_prediction(loan_amt,annual_inc,int_rate,tot_cur_bal) 


