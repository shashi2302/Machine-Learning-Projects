from flask import Flask, jsonify
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load trained model
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Load weights or use a pre-trained model here
    return model

keras_model = load_model()

@app.route("/predict/<user_id>")
def predict(user_id):
    user_data = pd.read_csv(f"data/user_{user_id}_data.csv")
    features = user_data[["heart_rate", "sleep_duration", "activity_level"]].values[-1:]
    prediction = keras_model.predict(features)
    return jsonify({"user_id": user_id, "prediction": float(prediction[0][0])})

if __name__ == "__main__":
    app.run(debug=True)
