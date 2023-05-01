import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

employee_df=pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

employee_df.drop(columns=['Over18','EmployeeCount','StandardHours','EmployeeNumber'],inplace=True)

employee_df['Attrition'] = employee_df['Attrition'].map({'Yes': 1, 'No': 0})
employee_df['OverTime'] = employee_df['OverTime'].map({'Yes': 1, 'No': 0})

employee_df['Total_Satisfaction'] = (employee_df['EnvironmentSatisfaction'] + 
                                     employee_df['JobInvolvement'] + 
                                     employee_df['JobSatisfaction'] + 
                                     employee_df['RelationshipSatisfaction'] +
                                     employee_df['WorkLifeBalance']) /5 

employee_df.drop(columns=['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], inplace=True)

employee_df['Total_Satisfaction_bool'] = employee_df['Total_Satisfaction'].apply(lambda x:1 if x>=2.2 else 0 ) 
employee_df.drop('Total_Satisfaction', axis=1, inplace=True)
employee_df['Age_bool'] = employee_df['Age'].apply(lambda x:1 if x<35 else 0)
employee_df.drop('Age', axis=1, inplace=True)
employee_df['DailyRate_bool'] = employee_df['DailyRate'].apply(lambda x:1 if x<750 else 0)
employee_df.drop('DailyRate', axis=1, inplace=True)
employee_df['Department_bool'] = employee_df['Department'].apply(lambda x:1 if x=='Sales' else 0)
employee_df.drop('Department', axis=1, inplace=True)
employee_df['DistanceFromHome_bool'] = employee_df['DistanceFromHome'].apply(lambda x:1 if x>10 else 0)
employee_df.drop('DistanceFromHome', axis=1, inplace=True)
employee_df['HourlyRate_bool'] = employee_df['HourlyRate'].apply(lambda x:1 if x<65 else 0)
employee_df.drop('HourlyRate', axis=1, inplace=True)
employee_df['JobRole_bool'] = employee_df['JobRole'].apply(lambda x:1 if x=='Sales Executive' else 0)
employee_df.drop('JobRole', axis=1, inplace=True)
employee_df['MonthlyIncome_bool'] = employee_df['MonthlyIncome'].apply(lambda x:1 if x<3500 else 0)
employee_df.drop('MonthlyIncome', axis=1, inplace=True)
employee_df['NumCompaniesWorked_bool'] = employee_df['NumCompaniesWorked'].apply(lambda x:1 if x>4 else 0)
employee_df.drop('NumCompaniesWorked', axis=1, inplace=True)
employee_df['TotalWorkingYears_bool'] = employee_df['TotalWorkingYears'].apply(lambda x:1 if x<8 else 0)
employee_df.drop('TotalWorkingYears', axis=1, inplace=True)
employee_df['YearsAtCompany_bool'] = employee_df['YearsAtCompany'].apply(lambda x:1 if x<3 else 0)
employee_df.drop('YearsAtCompany', axis=1, inplace=True)
employee_df['YearsInCurrentRole_bool'] = employee_df['YearsInCurrentRole'].apply(lambda x:1 if x<3 else 0)
employee_df.drop('YearsInCurrentRole', axis=1, inplace=True)
employee_df['YearsSinceLastPromotion_bool'] = employee_df['YearsSinceLastPromotion'].apply(lambda x:1 if x<1 else 0)
employee_df.drop('YearsSinceLastPromotion', axis=1, inplace=True)
employee_df['YearsWithCurrManager_bool'] = employee_df['YearsWithCurrManager'].apply(lambda x:1 if x<1 else 0)
employee_df.drop('YearsWithCurrManager', axis=1, inplace=True)
employee_df.drop('MonthlyRate', axis=1, inplace=True)
employee_df.drop('PercentSalaryHike', axis=1, inplace=True)
employee_df['Gender'] = employee_df['Gender'].apply(lambda x:1 if x=='Female' else 0)

convert_category = ['BusinessTravel','Education','EducationField','MaritalStatus','StockOptionLevel','OverTime','Gender','TrainingTimesLastYear']
for col in convert_category:
        employee_df[col] = employee_df[col].astype('category')
        
X_categorical = employee_df.select_dtypes(include=['category'])
X_numerical = employee_df.select_dtypes(include=['int64'])

y = employee_df['Attrition']

X_numerical.drop('Attrition', axis=1, inplace=True)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

X_categorical = onehotencoder.fit_transform(X_categorical).toarray()
X_categorical = pd.DataFrame(X_categorical)

X_all = pd.concat([X_categorical, X_numerical], axis=1)

X_all.columns = X_all.columns.astype(str)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression model created with accuracy of {:.2f}%".format(100* accuracy_score(y_pred, y_test)))

import pickle
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(onehotencoder,open('encoder.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))

print('----------------------------')
print('Model created sucessfully')
print('----------------------------')