import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns",None)
import plotly.express as px
import sklearn
import scipy
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

data = pd.read_csv("C:\\Users\\WalkingTree\\Desktop\\KC\\Attrition\\data\\data.csv")

# Step 1: Label Encoding for categorical variables
label_encoder = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = label_encoder.fit_transform(data[col])

# Step 2: One-Hot Encoding for categorical variables (apply directly to original columns)
df = pd.get_dummies(data, columns=data.select_dtypes(include='object').columns)

# Display the result
print(df)

#Splittling data in X & Y for training
x = df.drop(columns = ['Attrition'])
y = df['Attrition']

#Splitting the data in Train & Test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#Scaling the data
scaler = RobustScaler()
x_scaler = scaler.fit_transform(x)
x= x_scaler

#Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#Predicting the data
y_pred= model.predict(x_test)

print("Accuracy :", accuracy_score(y_test,y_pred))

#Saving the model
with open('models/attrition_model.pkl',"wb") as f:
    pickle.dump(model,f)

                   