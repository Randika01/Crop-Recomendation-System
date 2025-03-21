from flask import Flask, request, render_template,redirect, url_for, flash,jsonify, session
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

app = Flask(__name__)
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

#Admin table
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # Store hashed passwords

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
storage_model = load_model("lstm_tank_storage.h5", custom_objects={"mse": MeanSquaredError()})  # LSTM model for tank storage prediction
storage_scaler = joblib.load('storage_scaler.pkl')              # MinMaxScaler for Storage (%)
storage_encoder_district = joblib.load('water_encoder_range.pkl')                # OneHotEncoder for Range (District)
storage_encoder_month = joblib.load("water_encoder_month.pkl")  # LabelEncoder for Month

crop_model = load_model('crop_recommendation_model.h5')         # Model for crop recommendation
crop_scaler = joblib.load('crop_scaler.pkl')                    # MinMaxScaler for crop recommendation features
crop_encoder = joblib.load('crop_encoder.pkl')                  # LabelEncoder for crop names

#load temperature model and scalers 
temperature_model = load_model("temperature_prediction_model.h5")
temp_encoder_month = joblib.load("Temp_encoder_month.pkl")  # LabelEncoder for month
temp_encoder_district = joblib.load("Temp_encoder_district.pkl")  # LabelEncoder for district
temperature_scaler = joblib.load("temperature_scaler.pkl")  # Load the appropriate scaler

#rainfall  files
rainfall_model = load_model("rainfall_prediction_model.h5")
rainfall_encoder_month = joblib.load("rainfall_encoder_month.pkl")  # LabelEncoder for month
rainfall_encoder_district = joblib.load("rainfall_encoder_district.pkl")  # LabelEncoder for district
rainfall_scaler = joblib.load("rainfall_scaler.pkl")
# Load datasets
tank_data = pd.read_csv("Tank_Water_Storage.csv")               # Water tank storage dataset
tank_data['Month'] = pd.to_datetime(tank_data['Month'], format='%B').dt.month  # Convert month names to numeric

crop_data = pd.read_csv("Crop_recommendation.csv")              # Crop recommendation dataset

print("Scaler Min:", storage_scaler.data_min_)
print("Scaler Max:", storage_scaler.data_max_)

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
def get_counts():
    user_count = User.query.count()  # Use SQLAlchemy to count users
    crop_count = CropRecommendation.query.count()  # Use SQLAlchemy to count crops
    feedback_count = Feedback.query.count()  # Count of feedback
    return user_count, crop_count, feedback_count


@app.route('/admin')
def admin_dashboard():
    # Check if the user is logged in
    if 'username' not in session:
        return redirect(url_for('admin_login'))  # Redirect to login if not logged in

    user_count, crop_count,feedback_count  = get_counts()  # Get the data to display on the dashboard
    return render_template('admin/index.html', users=user_count, crops=crop_count, feedback_count=feedback_count)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Fetch the admin from the database based on the entered username
        admin = Admin.query.filter_by(username=username).first()

        if admin and check_password_hash(admin.password, password):  # Check the hashed password
            session['username'] = admin.username  # Store the username in the session
            return redirect(url_for('admin_dashboard'))  # Redirect to the admin dashboard
        else:
            flash('Invalid username or password', 'danger')  # Flash message for failed login
            return render_template('/admin/login.html')

    return render_template('/admin/login.html')

@app.route('/admin/signup', methods=['GET', 'POST'])
def admin_signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('admin_signup'))

        existing_user = Admin.query.filter_by(username=username).first()
        existing_email = Admin.query.filter_by(email=email).first()

        if existing_user:
            flash("Username already taken", "danger")
            return redirect(url_for('admin_signup'))
        if existing_email:
            flash("Email already registered", "danger")
            return redirect(url_for('admin_signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_admin = Admin(username=username, email=email, password=hashed_password)

        db.session.add(new_admin)
        db.session.commit()

        flash("Admin account created successfully!", "success")
        return redirect(url_for('admin_login'))

    return render_template('admin/login.html')  #  Correctly rendering admin login page


    
@app.route('/admin/logout')
def admin_logout():
    session.pop('username', None)  # Remove the username from the session
    return redirect(url_for('admin_login'))  # Redirect to login page
 
    
    #Route to display users
@app.route('/admin/user', methods=['GET'])
def user():
    users = User.query.all()
      # Fetching all users from the User table
    return render_template('admin/user.html', users=users)

@app.route('/admin/feedback', methods=['GET'])
def admin_feedback():
    try:
        feedbacks = Feedback.query.all()
        print("Fetched Feedback:", feedbacks)  # Debugging line
        return render_template("admin/feedback.html", feedbacks=feedbacks)
    except Exception as e:
        print("Error fetching feedback:", str(e))
        return "Error fetching feedback", 500
    
@app.route('/delete_feedback/<int:feedback_id>', methods=['POST'])
def delete_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    db.session.delete(feedback)
    db.session.commit()
    return redirect(url_for('admin_feedback'))

@app.route('/api/farmer_data')
def get_farmer_data():
    # Query to group by farmer type and calculate average farm size
    farmer_data = db.session.query(
        User.farmer_type,
        func.avg(User.farm_size).label('avg_farm_size'),
        func.count(User.id).label('farmer_count')
    ).group_by(User.farmer_type).all()

    # Format data for JSON response
    data = [{"farmer_type": row.farmer_type, "avg_farm_size": float(row.avg_farm_size)} for row in farmer_data]

    return jsonify(data)

# Route to delete a user
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get(user_id)  # Find the user by their ID
    if user:
        db.session.delete(user)  # Delete the user from the database
        db.session.commit()  # Commit the transaction to save changes
    return redirect(url_for('user'))
    
@app.route('/admin/crops')
def crops():
    # Fetch all crop recommendations from the database
    crops_data = CropRecommendation.query.all()
    return render_template('admin/crops.html', crops=crops_data)

@app.route('/delete_crop/<int:crop_id>', methods=['POST'])
def delete_crop(crop_id):
    # Find and delete the crop entry by ID from the database
    crop = CropRecommendation.query.get(crop_id)
    if crop:
        db.session.delete(crop)  # Delete the crop record from the database
        db.session.commit()  # Commit the transaction to save the changes
        flash('Crop deleted successfully.', 'success')
    else:
        flash('Crop not found.', 'danger')
    
    return redirect(url_for('crops'))

@app.route('/api/crop_recommendations')
def crop_recommendations():
    # Fetch district and recommended_crop data from CropRecommendation table
    crop_data = db.session.query(CropRecommendation.district, CropRecommendation.selected_crop).all()
    
    # Aggregate data to get the count of each crop by district
    district_crop_count = {}
    for district, crop in crop_data:
        if district not in district_crop_count:
            district_crop_count[district] = {}
        if crop not in district_crop_count[district]:
            district_crop_count[district][crop] = 0
        district_crop_count[district][crop] += 1
    
    # Prepare the data for the Pie Chart (total count of crops per district)
    chart_data = []
    for district, crops in district_crop_count.items():
        total_crops = sum(crops.values())  # Total crops in each district
        chart_data.append({
            'district': district,
            'total_crops': total_crops
        })
    
    return jsonify(chart_data)




@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        feedback_message = request.form['message']
        
        # Save feedback to the database
        feedback = Feedback(name=name, email=email, feedback=feedback_message)
        db.session.add(feedback)
        db.session.commit()
        
        # Redirect after successful submission (you can modify the redirect)
        return jsonify({"message": "Feedback submitted successfully!"}), 200
    return render_template('contact.html')


# Admin Panel to view feedbacks
@app.route('/admin')
def admin():
    # Query all feedback messages from the database
    feedbacks = Feedback.query.all()
    return render_template('admin_dashboard.html', feedbacks=feedbacks)

# @app.route('/map')
# def map_page():
#     return render_template('map.html')    map route

@app.route('/how-it-work')
def how_work():
    return render_template('how-it-works.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/about')
def about_page():
    return render_template('about.html')


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


#temperature and rainfall
@app.route('/predict_rainfall_and_temperature', methods=['POST'])
def predict_rainfall_and_temperature():
    """Predict both rainfall and temperature based on district and month"""
    
    district = request.form['district']
    month = request.form['month']
    predicted_storage = session.get('predicted_storage')

    # Predict Rainfall
    encoded_month_rainfall = rainfall_encoder_month.transform([month])[0]
    encoded_district_rainfall = rainfall_encoder_district.transform([district])[0]
    features_rainfall = np.array([[encoded_month_rainfall, encoded_district_rainfall]])
    features_rainfall_scaled = rainfall_scaler.transform(features_rainfall)
    predicted_rainfall = rainfall_model.predict(features_rainfall_scaled)

    # Predict Temperature
    encoded_month_temp = temp_encoder_month.transform([month])[0]
    encoded_district_temp = temp_encoder_district.transform([district])[0]
    features_temp = np.array([[encoded_month_temp, encoded_district_temp]])
    features_temp_scaled = temperature_scaler.transform(features_temp)
    predicted_temperature = temperature_model.predict(features_temp_scaled)

    # Convert NumPy array to Python float before storing in session
    session['predicted_rainfall'] = float(predicted_rainfall.item())  
    session['predicted_temperature'] = float(predicted_temperature.item())  


    # Show the predictions
    return render_template(
        'crop_prediction.html',
        rainfall_statement=f"Predicted Rainfall: {predicted_rainfall[0]} mm",
        temperature_statement=f"Predicted Temperature: {predicted_temperature[0]}°C",
        predicted_storage=predicted_storage,
        district=district,
        month=month
    )
    
@app.route('/crop_prediction')
def crop_prediction():
    # Your logic for rendering the crop prediction page
    return render_template('crop_prediction.html')

@app.route("/predict_storage", methods=['POST'])
def predict_storage():
    # Step 1: Predict water storage based on user input (District & Month)
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
    predicted_storage = storage_scaler.inverse_transform(predicted_storage_normalized.reshape(-1, 1))[0][0]

    print(f"Raw model prediction: {predicted_storage_normalized}")
    print(f"Inverse transformed prediction: {predicted_storage}")
    
    # storage sufficiency decision
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
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    
    rainfall = float(session.get('predicted_rainfall', 0))
    temperature = float(session.get('predicted_temperature', 0))

    
    # Retrieve the predicted storage sufficiency from storage prediction logic
    predicted_storage = float(request.form.get('Predict_Storage', 0))

    # Adjust rainfall if storage is sufficient
    if predicted_storage > 70:
       predicted_rainfall = crop_data['Rainfall'].mean()  # Replace rainfall with the mean value

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
