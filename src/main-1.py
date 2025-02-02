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
import pickle


def train_model():
    #Load Data
    data_path = "C:\\Users\\WalkingTree\\Desktop\KC\Attrition\data\data.csv"

    if not os.path.exists(data_path):
        print(f'Error: {data_path} does not exist')
        return
    
    data = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    for col in data.selct_dtypes(include = 'object').columns:
        data[col] = label_encoder.fit_transform(data[col])

    df = pd.get_dummies(data,columns = data.select_dtypes(include = 'object').columns)

    X = df.drop('Attrition'),axis =1
    y = df['Attrition']

     # Splitting into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test == scaler.transform(X_test)

    #Model Training:

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    #save the model and scaler
    os.makedirs("models", exist_ok = True)
    with open("models/attrition_model.pkl","wb") as f:
        pickle.dump(model,f)

    with open("models/scaler.pkl","wb") as f:
        pickle.dump(scaler,f)

if __name__ =="__main__":
    train_model()






                   
