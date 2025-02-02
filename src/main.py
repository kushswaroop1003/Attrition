import os
import pickle
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score
from typing import Tuple

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

#Utility Functions:

def load_data(data_path:str)->pd.DataFrame:
    "Load data from given path"

    if not os.path.exists(data_path):
        return FileNotFoundError(f"Error: {data_path} does not exist.")
    return pd.read_csv(data_path)

def preprocess_data(data:pd.DataFrame) -> Tuple[pd.DataFrame,pd.Series]:
    """Preprocess the data: encode categorical variables and split features/target."""

    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include = "object").columns:
        data[col] = label_encoder.fit_transform(data[col])

    df = pd.get_dummies(data,columns = data.select_dtypes(include = 'object').columns)
    print("------")
    print(df)
    print("-----------")
    # Ensure target column exists
    target_column = "Attrition"
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in the dataset")
    
    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df["Attrition"]

    return X,y

def scale_data (X_train:pd.DataFrame,X_test: pd.DataFrame) ->Tuple[pd.DataFrame,pd.DataFrame,RobustScaler]:
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled,scaler


def save_model_and_scaler(model, scaler, model_dir: str = "models"):
    """Save the trained model and scaler to the specified directory."""
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "attrition_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

def train_model(data_path: str):
    "Main Function to Train the Model"

    try:
        print("Hii")

        data = load_data(data_path)
        print("HII Again")
        X,y = preprocess_data(data)
        print("HII Once Again")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled,y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        save_model_and_scaler(model, scaler)
    except Exception as e:
        print(f"Error occurred during training: {e}")

if __name__ == "__main__":
    data_path = r"C:/Users/WalkingTree/Desktop/KC/Attrition/data/data.csv"
    train_model(data_path)
