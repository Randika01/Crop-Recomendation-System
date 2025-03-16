from flask import Flask, request, render_template,redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import joblib
from flask_migrate import Migrate
from flask_cors import CORS
from tensorflow.keras.losses import MeanSquaredError
from sqlalchemy import func
from datetime import datetime

app = Flask(__name__, template_folder='.')
CORS(app) 

CROP_IMAGES ={
    "rice":"crop-images/rice.jpg",
    "maize":"crop-images/maize.jpg",
    "chickpea":"crop-images/Chickpea.jpg",
    "kidneybeans":"crop-images/kidneybeans.jpg",
    "pigeonpeas":"crop-images/pigeonpeas.jpg",
    "mothbeans":"crop-images/mothbeans.jpg",
    "mungbean":"crop-images/mungbeans.jpg",
    "blackgram":"crop-images/blackgram.jpg",
    "lentil":"crop-images/Lentil.jpg",
    "pomegranate":"crop-images/pomegranate.jpg",
    "banana":"crop-images/banana.jpg",
    "mango":"crop-images/mango.jpg",
    "grapes":"crop-images/grapes.jpg",
    "watermelon":"crop-images/watermelon.jpg",
    "muskmelon":"crop-images/muskmelon.jpg",
    "apple":"crop-images/apple.jpg",
    "orange":"crop-images/orange.jpg",
    "papaya":"crop-images/papaya.jpg",
    "coffee":"crop-images/coffe.jpg",
    "coconut":"crop-images/Coconut.jpg",
    "cotton":"crop-images/cotton.jpg"
}

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
    selected_crop = db.Column(db.String(100), nullable=True)

# Feedback Model
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Feedback {self.id} by {self.name}>'
    

# Create the database tables
with app.app_context():
    db.create_all()

# Load models and scalers
storage_model = load_model('single_lstm_storage_predictor.h5', custom_objects={"mse": MeanSquaredError()})  # LSTM model for tank storage prediction
storage_scaler = joblib.load('storage_scaler.pkl')              # MinMaxScaler for Storage (%)
storage_encoder_district = joblib.load('water_encoder_range.pkl')                # OneHotEncoder for Range (District)
storage_encoder_month = joblib.load("water_encoder_month.pkl")  # LabelEncoder for Month

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
    return render_template("index.html", username=session.get('username'))

@app.route("/save_crop", methods=['POST'])
def save_crop():
    selected_crop = request.form['selected_crop']
    crop_id = request.form['crop_id']

    # Find the crop recommendation record by ID
    recommendation = CropRecommendation.query.get(crop_id)
    
    if recommendation:
        # Update the selected crop for the user
        recommendation.selected_crop = selected_crop
        db.session.commit()
        flash(f'Your selected crop "{selected_crop}" has been saved successfully!', 'success')
    else:
        flash('Error saving selected crop. Please try again.', 'danger')

    return redirect(url_for('crop_prediction'))

@app.route('/crop_prediction')
def crop_prediction():
    # Your logic for rendering the crop prediction page
    return render_template('crop_prediction.html')

@app.route("/predict_storage", methods=['POST'])
def predict_storage():
    # Step 1: Predict tank storage based on Area and Month
    range_selected = request.form['district']
    month_name = request.form['month']

    # Define districts without tanks
    no_tank_districts = ["Colombo", "Gampaha", "Jaffna", "Kaluthara", "Kegalle","Kilinochchi", "Matara","Matale","Vavuniya", "Mullativu", "Nuwara Eliya", "Ratnapura"]

    if range_selected in no_tank_districts:
        storage_statement = f"No tanks available in {range_selected}."
        session['storage_statement'] = storage_statement
        return render_template(
            "crop_prediction.html",
            storage_statement=storage_statement,
            predicted_storage=None,
            district=range_selected,
            month=month_name,
        )
    
    # Encode categorical values
    month_encoded = storage_encoder_month.transform([month_name])[0]
    range_encoded = storage_encoder_district.transform([range_selected])[0]

    # Prepare input for the model
    sample_input = np.array([[month_encoded, range_encoded]]).reshape(1, 1, 2)


    # Predict storage for the selected month
    predicted_storage_normalized = storage_model.predict(sample_input)
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
        
    # Store in session
    session['storage_statement'] = storage_statement

    return render_template(
        "crop_prediction.html",
        storage_statement=storage_statement,
        predicted_storage=predicted_storage, district=range_selected,
        month=month_name
    )

@app.route("/predict_crop", methods=['POST'])
def predict_crop():
    # Step 2: Predict crop based on soil nutrients and weather data
    district = request.form.get('district')  # Retrieve district
    month = request.form.get('month')        # Retrieve month
    
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    
    rainfall = float(session.get('predicted_rainfall', 0))
    temperature = float(session.get('predicted_temperature', 0))

    
    # Retrieve the predicted storage sufficiency from storage prediction logic
    predicted_storage = float(request.form.get('Predict_Storage', 0))

    # Adjust rainfall if storage is sufficient
    if predicted_storage > 70:
        rainfall = crop_data['Rainfall'].mean()  # Replace rainfall with the mean value

    # Prepare features for crop recommendation
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = crop_scaler.transform(features)

    # Predict crop
    crop_prediction = crop_model.predict(features_scaled)
    # Get the top 4 crops based on predicted probabilities
    top_4_crop_indices = np.argsort(crop_prediction[0])[-4:][::-1]  # Get the top 4 indices
    top_4_crop_names = crop_encoder.inverse_transform(top_4_crop_indices)  # Get the corresponding crop names

    top_4_crop_images = [CROP_IMAGES.get(crop, "default.jpg") for crop in top_4_crop_names]
    # Save the recommendation to the database
    if 'user_id' in session:
        user_id = session['user_id']
        recommendation = CropRecommendation(
            user_id=user_id,
            nitrogen=N,
            phosphorus=P,
            potassium=K,
            temperature=temperature,
            humidity=humidity,
            pH=ph,
            rainfall=rainfall,
            district=district,
            month=month
        )
        db.session.add(recommendation)
        db.session.commit()
    
    return render_template(
       "crop_prediction.html",
        crop_result=f"The recommended crops are:",
        crop_names=top_4_crop_names,
        crop_images=top_4_crop_images,
        crop_id=recommendation.id 
    )
    
@app.route("/recommendations")
def recommendations():
    if 'user_id' not in session:
        flash('Please log in to view your recommendations.', 'info')
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    recommendations = CropRecommendation.query.filter_by(user_id=user_id).all()
    
    return render_template("recommendations.html", recommendations=recommendations)

@app.route("/saved_crops",  methods=['GET'])
def saved_crops():
    try:
        crops_by_district = db.session.query(
            CropRecommendation.district,
            CropRecommendation.month,
            CropRecommendation.selected_crop
        ).filter(CropRecommendation.selected_crop.isnot(None)).all()

        print("Crops by District (Debugging):", crops_by_district)

        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }

        district_centers = {
        "Ampara": [81.6726, 7.2915],
        "Anuradhapura": [80.4036, 8.3114],
        "Badulla": [81.0550, 6.9898],
        "Batticalo": [81.6822, 7.7144],
        "Colombo": [79.8612, 6.9271],
        "Galle": [80.2170, 6.0535],
        "Gampaha": [79.9981, 7.0914],
        "Hambantota": [81.1240, 6.1241],
        "Jaffna": [80.0144, 9.6615],
        "Kalutara": [79.9594, 6.5854],
        "Kandy": [80.6350, 7.2906],
        "Kegalle": [80.3478, 7.2522],
        "Kilinochchi": [80.4181, 9.3803],
        "Kurunegala": [80.3728, 7.4865],
        "Mannar": [79.9045, 8.9761],
        "Matale": [80.7323, 7.4679],
        "Matara": [80.5353, 5.9488],
        "Monaragala": [81.3483, 6.8712],
        "Mullativu": [80.8298, 9.2800],
        "Nuwara Eliya": [80.7812, 6.9497],
        "Polonnaruwa": [81.0011, 7.9409],
        "Puttalam": [79.8398, 8.0362],
        "Ratnapura": [80.4012, 6.6828],
        "Trincomalee": [81.2336, 8.5874],
        "Vavuniya": [80.4928, 8.7540]
        }


        for district, month, selected_crop in crops_by_district:
            coordinates = district_centers.get(district, [80.7718, 7.8731])  # Fallback coordinates
            feature = {
                "type": "Feature",
                "properties": {
                    "name": district,
                    "month": month,
                    "selected_crop": selected_crop
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates
                }
            }
            geojson_data["features"].append(feature)

        return geojson_data

    except Exception as e:
            app.logger.error(f"Error in /saved_crops: {e}")
            return {"error": "Failed to load crops data"}, 500

if __name__ == "__main__":
    app.run(debug=True)
