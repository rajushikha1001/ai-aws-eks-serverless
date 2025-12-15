import json
import joblib

model = joblib.load("/opt/model/model.pkl")

def handler(event, context):
    features = event["features"]
    prediction = model.predict([features])
    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": float(prediction[0])})
    }
