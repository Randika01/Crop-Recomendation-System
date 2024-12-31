from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import joblib

app = Flask(__name__, template_folder='.')

# Load models and scalers
storage_model = load_model('single_lstm_storage_predictor.h5')  # LSTM model for tank storage prediction
storage_scaler = joblib.load('storage_scaler.pkl')              # MinMaxScaler for Storage (%)
range_encoder = joblib.load('range_encoder.pkl')                # OneHotEncoder for Range (District)

crop_model = load_model('crop_recommendation_model.h5')         # Model for crop recommendation
crop_scaler = joblib.load('crop_scaler.pkl')                    # MinMaxScaler for crop recommendation features
crop_encoder = joblib.load('crop_encoder.pkl')                  # LabelEncoder for crop names

# Load datasets
tank_data = pd.read_csv("Tank_Water_Storage.csv")               # Water tank storage dataset
tank_data['Month'] = pd.to_datetime(tank_data['Month'], format='%B').dt.month  # Convert month names to numeric

crop_data = pd.read_csv("Crop_recommendation.csv")              # Crop recommendation dataset

@app.route("/")
def index():
    return render_template("crop_prediction.html")

@app.route("/predict_storage", methods=['POST'])
def predict_storage():
    # Step 1: Predict tank storage based on Area and Month
    range_selected = request.form['Area']
    month_name = request.form['Month']

    # Map month names to numeric values
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    month_selected = month_mapping[month_name]

    # Filter data for the selected district
    filtered_data = tank_data[tank_data['Range'] == range_selected].sort_values(by=['Year', 'Month'])
    recent_storage = filtered_data['Storage (%)'].values[-12:].reshape(-1, 1)

    # One-hot encode the selected range
    range_encoded = range_encoder.transform([[range_selected]])
    input_sequence = []
    for storage in recent_storage:
        input_sequence.append(np.concatenate((storage, range_encoded[0])))

    input_sequence = np.array(input_sequence).reshape(1, 12, -1)

    # Predict storage for the selected month
    predicted_storage_normalized = storage_model.predict(input_sequence)
    predicted_storage = storage_scaler.inverse_transform(predicted_storage_normalized)[0][0]

    # Determine if storage is sufficient
    if predicted_storage > 70:
        storage_statement = (f"The predicted tank storage for {range_selected} in {month_name} is "
                             f"{predicted_storage:.2f}%. The storage is sufficient, and you can use tank water "
                             f"for your cultivation.")
    else:
        storage_statement = (f"The predicted tank storage for {range_selected} in {month_name} is "
                             f"{predicted_storage:.2f}%. The storage is insufficient, and rainfall will play "
                             f"a critical role in crop cultivation.")

    return render_template(
        "crop_prediction.html",
        storage_statement=storage_statement,
        predicted_storage=predicted_storage
    )

@app.route("/predict_crop", methods=['POST'])
def predict_crop():
    # Step 2: Predict crop based on soil nutrients and weather data
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])  # User-entered rainfall

    # Retrieve the predicted storage sufficiency from storage prediction logic
    predicted_storage = float(request.form.get('Predict_Storage', 0))

    # Adjust rainfall if storage is sufficient
    if predicted_storage > 70:
        rainfall = crop_data['Rainfall'].mean()  # Replace rainfall with the mean value

    # Prepare features for crop recommendation
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    features_scaled = crop_scaler.transform(features)

    # Predict crop
    crop_prediction = crop_model.predict(features_scaled)
    crop_index = np.argmax(crop_prediction)
    crop_name = crop_encoder.inverse_transform([crop_index])[0]

    return render_template(
        "crop_prediction.html",
        crop_result=f"The recommended crop is: {crop_name}"
    )

if __name__ == "__main__":
    app.run(debug=True)
