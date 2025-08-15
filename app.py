from flask import Flask, render_template, request, jsonify
import pandas as pd
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

FEATURES = model.get_booster().feature_names

app = Flask(__name__)


def preprocess(data):
    df = pd.DataFrame([data])
    df['Gender_encoded'] = 1 if data['Gender'] == 'male' else 0
    df['Gender_male'] = 1 if data['Gender'] == 'male' else 0
    df['Gender_female'] = 1 if data['Gender'] == 'female' else 0
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['HR_Zone_Low'] = int(df['Heart_Rate'].values[0] < 100)
    df['HR_Zone_Moderate'] = int(100 <= df['Heart_Rate'].values[0] < 140)
    df['Temp_Normal'] = int(df['Body_Temp'].values[0] < 37.5)
    df['Temp_High'] = int(df['Body_Temp'].values[0] >= 37.5)
    df['Age_Group_Young'] = int(df['Age'].values[0] < 30)
    df['Age_Group_Middle'] = int(30 <= df['Age'].values[0] < 60)
    df['Age_Group_Senior'] = int(df['Age'].values[0] >= 60)
    df['Weight_HR'] = df['Weight'] * df['Heart_Rate']
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['HR_Duration'] = df['Heart_Rate'] * df['Duration']
    return df[FEATURES]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = preprocess(data)

    # Reorder features to match training order exactly
    df = df[FEATURES]  # <- make sure this list is in the same order as during training

    prediction = model.predict(df)[0]
    return jsonify({"calories": round(float(prediction), 2)})


if __name__ == "__main__":
    app.run(debug=True)
app.run(host='0.0.0.0', port=5000)
