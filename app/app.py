import os
import pickle
import pandas as pd
from flask import Flask, request,jsonify
from sklearn.preprocessing import RobustScaler
from dotenv import load_dotenv

load_dotenv()

#initialize the app
app = Flask(__name__)

#Load Model Path
Model_Path = os.environ.get('MODEL_PATH')
Scaler_Path = os.environ.get('SCALER_PATH')

if not os.path.exists(Model_Path) or not os.path.exists(Scaler_Path):
    raise FileNotFoundError ("Model or Scaler file not found. Train the model first.")

with open(Model_Path,"rb") as f:
    model = pickle.load(f)

with open(Scaler_Path,"rb") as f:
    scaler = pickle.load(f)

@app.route("/predict",methods =["POST"])

def predict():
    try:
        #Get json data from request

        data = request.get_json()
        print(data)
        if not data:
            return jsonify({"error: no data provided"}),400
        
        #convert JSON to dataframe
        input_df = pd.DataFrame(data)

        #scaled data
        input_scaled = scaler.transform(input_df)

        #predict
        predictions = model.predict(input_scaled)

        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = True)