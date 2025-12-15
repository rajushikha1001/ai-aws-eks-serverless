from app.model import model

def predict(features):
    prediction = model.predict([features])
    return float(prediction[0])


