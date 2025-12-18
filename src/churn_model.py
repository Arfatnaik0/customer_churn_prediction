# import necessary libraries
import numpy as np
import pandas as pd

# load dataset
churn = pd.read_csv('..\\data\\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# data cleaning and preprocessing
churn['TotalCharges']=pd.to_numeric(churn['TotalCharges'],errors='coerce')
churn['TotalCharges']=churn['TotalCharges'].fillna(churn['TotalCharges'].median())
churn.drop('customerID',axis=1,inplace=True)

# replace 'No phone service' and 'No internet service' with 'No'
replace_dict = {'No phone service': 'No', 'No internet service': 'No'}
cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in cols:
    churn[col] = churn[col].replace(replace_dict)


# import ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# x and y split
x=churn.drop('Churn', axis=1)
y=churn['Churn'].map({'Yes': 1, 'No': 0})

# identify numerical and categorical columns
num_cols=['tenure','MonthlyCharges','TotalCharges']
cat_cols=[column for column in x.columns if column not in num_cols]

# preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# complete pipeline with logistic regression
model=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced'))
])

# train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# fit the model
model.fit(x_train,y_train)
y_prob=model.predict_proba(x_test)[:,1]
y_pred=(y_prob>0.4).astype(int)

# metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(f'ROC-AUC score:{roc_auc_score(y_test,y_prob)}')

# save the model
joblib.dump(model,'..\\model\\churn_model.pkl')




