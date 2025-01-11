from flask import Flask, request, render_template,redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import joblib
from flask_migrate import Migrate
from flask_cors import CORS

app = Flask(__name__, template_folder='.')
CORS(app) 

# Set up the database URI and secret key for sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '3bb08d1b6b92e84572425b0d7d06acb2448cbb5c6ca54412'

# Initialize the database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    farmer_type = db.Column(db.String(50), nullable=False)
    province = db.Column(db.String(50), nullable=False)
    farm_size = db.Column(db.Float, nullable=False)

# Define the CropRecommendation model
class CropRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key to the User table
    district = db.Column(db.String(100), nullable=True)  # Update or add this line
    month = db.Column(db.String(20), nullable=True)  # Add this line
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    pH = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    recommended_crop = db.Column(db.String(100), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

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

# User Routes
@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form['full_name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        farmer_type = request.form['farmer_type']
        province = request.form['province']
        farm_size = request.form['farm_size']

        # Password hashing
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))
        
        # Use the correct hashing method
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Create a new user record
        new_user = User(full_name=full_name, phone=phone, email=email, password=hashed_password,
                        farmer_type=farmer_type, province=province, farm_size=farm_size)

        try:
            # Add to the database and commit
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except:
            flash('Error in account creation. Try again!', 'danger')
            return redirect(url_for('signup'))
    return render_template("signup.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store the user id in the session
            session['username'] = user.full_name  # Store the user's name in session
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  # Redirect to the main page
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            return redirect(url_for('crop_prediction'))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('user_id', None)  # Remove user_id from session
    session.pop('username', None)  # Remove username from session
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))
    
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
        
    )
    
