import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN to suppress TensorFlow messages

from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
from groq import Groq
from dotenv import load_dotenv
from twilio.rest import Client
import random
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
import datetime
from functools import wraps
from bson import ObjectId
import logging
from dateutil.parser import parse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import uuid
import joblib
import requests
import math
import smtplib
import jwt
from email.mime.text import MIMEText

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")
app.config["UPLOAD_FOLDER"] = "Uploads"
app.config["RESOLVED_FOLDER"] = "resolved_images"

# Create upload folders
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESOLVED_FOLDER"], exist_ok=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
twilio_sid = os.getenv("TWILIO_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone = os.getenv("TWILIO_PHONE")

# Validate environment variables
if not groq_api_key:
    logger.error("GROQ_API_KEY is not set in .env file")
    raise ValueError("GROQ_API_KEY is required")
if not all([twilio_sid, twilio_auth_token, twilio_phone]):
    logger.error("Twilio credentials are not set in .env file")
    raise ValueError("Twilio credentials (TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE) are required")

# Initialize Twilio client
try:
    client_twilio = Client(twilio_sid, twilio_auth_token)
    logger.info("Twilio client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Twilio client: {str(e)}")
    raise

# Initialize MongoDB client
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["WaterIssuesDB"]
    water_reports_collection = db["WaterReports"]
    users_collection = db["Users"]
    officers_collection = db["Officers"]
    water_quality_collection = db["WaterQualityPredictions"]
    comments_collection = db["report_comments"]
    flow_optimizations_collection = db["FlowOptimizations"]
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Initialize Groq client
try:
    groq_client = Groq(api_key=groq_api_key)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise

# Load TensorFlow model
model_path = "water_cnn_model.h5"
if not os.path.exists(model_path):
    logger.error(f"TensorFlow model file not found at {model_path}")
    raise FileNotFoundError(f"Model file {model_path} is missing")
try:
    water_model = tf.keras.models.load_model(model_path)
    logger.info("TensorFlow model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load TensorFlow model: {str(e)}")
    raise

# Load water quality prediction model
quality_model_path = "water_quality_model.pkl"
if not os.path.exists(quality_model_path):
    logger.error(f"Water quality model file not found at {quality_model_path}")
    raise FileNotFoundError(f"Model file {quality_model_path} is missing")
try:
    water_quality_model = joblib.load(quality_model_path)
    logger.info("Water quality model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load water quality model: {str(e)}")
    raise

otp_cache = {}

# Train predictive model
def train_predictive_model():
    try:
        reports = list(water_reports_collection.find({}, {"latitude": 1, "longitude": 1, "status": 1}))
        if len(reports) < 2:
            logger.warning("Insufficient reports for model training")
            return None, None
        X = [[r["latitude"], r["longitude"]] for r in reports]
        y = [1 if r["status"] in ["leakage", "pollution", "scarcity"] else 0 for r in reports]
        if len(set(y)) < 2:
            logger.warning("Data contains only one class, cannot train model")
            return None, None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression()
        model.fit(X_scaled, y)
        logger.info("Predictive model trained successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error training predictive model: {str(e)}")
        return None, None

# Retrain predictive model
def retrain_predictive_model():
    global predictive_model, scaler
    predictive_model, scaler = train_predictive_model() or (None, None)
    if predictive_model:
        logger.info("Predictive model retrained successfully")
    else:
        logger.warning("Predictive model retraining failed")

predictive_model, scaler = train_predictive_model() or (None, None)

# Token verification decorators
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("x-access-token")
        if not token:
            logger.error("Token is missing")
            return jsonify({"error": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_officer = officers_collection.find_one({"email": data["email"]})
            if not current_officer:
                logger.error("Officer not found for token")
                return jsonify({"error": "Invalid Token!"}), 401
        except Exception as e:
            logger.error(f"Invalid Token: {str(e)}")
            return jsonify({"error": "Invalid Token!"}), 401
        return f(current_officer, *args, **kwargs)
    return decorated

def user_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("x-access-token")
        if not token:
            logger.error("Token is missing")
            return jsonify({"error": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = users_collection.find_one({"phone": data["phone"]})
            if not current_user:
                logger.error("User not found for token")
                return jsonify({"error": "Invalid Token!"}), 401
        except Exception as e:
            logger.error(f"Invalid Token: {str(e)}")
            return jsonify({"error": "Invalid Token!"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Helper function to calculate distance between two coordinates (Haversine formula)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# Assign officer
def assign_officer():
    try:
        officer = officers_collection.find_one({}, sort=[("assigned_reports", 1)])
        if officer:
            officers_collection.update_one({"_id": officer["_id"]}, {"$inc": {"assigned_reports": 1}})
            logger.info(f"Officer assigned: {officer.get('name', 'Unknown')}")
            return officer
        logger.warning("No officers available")
        return None
    except Exception as e:
        logger.error(f"Error assigning officer: {str(e)}")
        return None

# Initialize WaterReports schema
def initialize_water_reports_schema():
    try:
        water_reports_collection.update_many(
            {"status": {"$exists": False}},
            {
                "$set": {
                    "status": "Pending",
                    "progress": 0,
                    "progress_notes": "",
                    "progress_image": None
                }
            }
        )
        logger.info("WaterReports schema initialized with new fields")
    except Exception as e:
        logger.error(f"Error initializing WaterReports schema: {str(e)}")

# Send notification (SMS and Email) to user
def send_notification(report_id, status):
    try:
        report = water_reports_collection.find_one({"_id": ObjectId(report_id)})
        if not report or "user_phone" not in report:
            logger.error(f"No user_phone found for report {report_id}")
            return
        user = users_collection.find_one({"phone": report["user_phone"]})
        if not user:
            logger.error(f"User not found for phone {report['user_phone']}")
            return

        message = f"Water issue report at {report['address']} is now {status}."
        if status.startswith("In-Progress"):
            message += f" Progress: {report.get('progress', 0)}%. Notes: {report.get('progress_notes', 'None')}."

        try:
            client_twilio.messages.create(
                body=message[:1600],
                from_=twilio_phone,
                to=user["phone"]
            )
            logger.info(f"SMS sent to {user['phone']} for report {report_id}: {status}")
        except Exception as e:
            logger.error(f"Failed to send SMS to {user['phone']} for report {report_id}: {str(e)}")

        try:
            msg = MIMEText(message)
            msg["Subject"] = "Water Issue Report Update"
            msg["From"] = os.getenv("SMTP_EMAIL", "no-reply@watermonitoring.com")
            msg["To"] = user["email"]
            with smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"), os.getenv("SMTP_PORT", 587)) as server:
                server.starttls()
                server.login(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
                server.send_message(msg)
            logger.info(f"Email sent to {user['email']} for report {report_id}: {status}")
        except Exception as e:
            logger.error(f"Failed to send email to {user['email']} for report {report_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in send_notification for report {report_id}: {str(e)}")

# Predictive maintenance endpoint
@app.route("/predict_maintenance", methods=["GET"])
@token_required
def predict_maintenance(current_officer):
    logger.info("Received predict_maintenance request")
    try:
        if not predictive_model or not scaler:
            logger.error("Predictive model not trained")
            return jsonify({"error": "Predictive model not available"}), 503
        reports = list(water_reports_collection.find(
            {"created_at": {"$gte": datetime.datetime.utcnow() - datetime.timedelta(days=30)}},
            {"latitude": 1, "longitude": 1, "address": 1, "status": 1}
        ))
        if not reports:
            logger.info("No recent reports for prediction")
            return jsonify({"predictions": []})
        X = [[r["latitude"], r["longitude"]] for r in reports]
        X_scaled = scaler.transform(X)
        probs = predictive_model.predict_proba(X_scaled)[:, 1]
        predictions = []
        for i, prob in enumerate(probs):
            if prob > 0.7:
                report = reports[i]
                predictions.append({
                    "latitude": report["latitude"],
                    "longitude": report["longitude"],
                    "address": report["address"],
                    "status": report["status"],
                    "risk_score": round(float(prob), 2)
                })
                try:
                    client_twilio.messages.create(
                        body=f"High-risk {report['status']} issue predicted at {report['address']}. Risk score: {round(prob, 2)}",
                        from_=twilio_phone,
                        to=current_officer["phone"]
                    )
                    logger.info(f"SMS sent to {current_officer['phone']} for {report['address']}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {current_officer['phone']}: {str(e)}")
        logger.info(f"Returning {len(predictions)} maintenance predictions")
        return jsonify({"predictions": predictions})
    except Exception as e:
        logger.error(f"Error in predict_maintenance: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Public heatmap data endpoint
@app.route("/map_data", methods=["GET"])
def get_map_data():
    logger.info("Received map_data request")
    try:
        status = request.args.get("status")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        query = {}
        if status:
            query["status"] = status
        if start_date and end_date:
            query["created_at"] = {
                "$gte": parse(start_date),
                "$lte": parse(end_date)
            }
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {
                        "latitude": "$latitude",
                        "longitude": "$longitude",
                        "status": "$status"
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "latitude": "$_id.latitude",
                    "longitude": "$_id.longitude",
                    "status": "$_id.status",
                    "count": 1,
                    "_id": 0
                }
            }
        ]
        heatmap_data = list(water_reports_collection.aggregate(pipeline))
        logger.info(f"Returning {len(heatmap_data)} heatmap data points")
        return jsonify(heatmap_data)
    except Exception as e:
        logger.error(f"Error fetching map data: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Public chatbot endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    logger.info("Received chatbot request")
    try:
        data = request.json
        user_message = data.get("message")
        if not user_message:
            logger.error("No message provided")
            return jsonify({"error": "No message provided"}), 400
        system_prompt = (
            "You are a helpful assistant for a water reporting system. Assist users with queries about water issues "
            "(e.g., leaks, pollution, scarcity), reporting processes, or system navigation. Provide clear, concise, "
            "and accurate responses. Guide users on how to log in, register, or view their reports as follows:\n"
            "- To register, visit the '/register' page and provide your name, phone, email, address, Aadhar, and password.\n"
            "- To log in, visit the '/login' page and enter your email or phone number along with your password.\n"
            "- To view your submitted reports, visit the '/reports' page after logging in.\n"
            "- To view a public heatmap of water issues, visit the '/map' page.\n"
            "- To report a water issue, provide an image of the issue along with your location (latitude and longitude) and a brief description.\n"
            "You are not allowed to access any user-specific data or personal information. "
            "Do not assume access to user-specific data unless explicitly provided in the query."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )
        response = chat_completion.choices[0].message.content
        logger.info(f"Chatbot response: {response}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Predict water issue
@app.route("/predict_water_issue", methods=["POST"])
@user_token_required
def predict_water_issue(current_user):
    logger.info("Received predict_water_issue request")
    if "image" not in request.files:
        logger.error("No image file provided")
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files["image"]
    lat = request.form.get("latitude")
    lng = request.form.get("longitude")
    address = request.form.get("address")
    if not lat or not lng or not address:
        logger.error("Location data missing")
        return jsonify({"error": "Location data missing"}), 400
    try:
        report_id = str(ObjectId())
        filename = f"{report_id}_water_issue.jpg"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(file_path)
        image_url = f"http://localhost:5000/uploads/{filename}"
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        preds = water_model.predict(img_array)[0]
        max_idx = np.argmax(preds)
        categories = ['leakage', 'pollution', 'scarcity']
        category = categories[max_idx] if preds[max_idx] >= 0.6 else "unknown"
        confidence = float(preds[max_idx]) if preds[max_idx] >= 0.6 else 0.0
        valid = confidence >= 0.6

        # Check if the category is pollution
        if category == "pollution":
            logger.info(f"Pollution detected, reporting as 'others' and not submitting report")
            return jsonify({
                "prediction": "others",
                "confidence": round(confidence, 2),
                "valid": valid,
                "latitude": lat,
                "longitude": lng,
                "address": address,
                "message": "Pollution issues are reported as 'others' and not submitted."
            }), 200

        assigned_officer_name = "No available officer"
        officer_email = None
        officer_phone = None
        officer = assign_officer()
        if officer and "name" in officer:
            assigned_officer_name = officer["name"]
            officer_email = officer.get("email", "N/A")
            officer_phone = officer.get("phone", "N/A")
        report_data = {
            "user_phone": current_user["phone"],
            "latitude": float(lat),
            "longitude": float(lng),
            "address": address,
            "status": category,
            "confidence": round(confidence, 2),
            "assigned_officer": assigned_officer_name,
            "officer_email": officer_email,
            "officer_phone": officer_phone,
            "image": image_url,
            "created_at": datetime.datetime.utcnow(),
            "resolved": False,
            "upvotes": 0,
            "upvoted_by": [],
            "status": "Pending",
            "progress": 0,
            "progress_notes": "",
            "progress_image": None
        }
        water_reports_collection.insert_one(report_data)
        # Send SMS to user to confirm report submission
        user_phone = current_user.get("phone")
        if user_phone:
            try:
                client_twilio.messages.create(
                    body=f"Your water issue report (ID: {report_id}) for {category} at {address} has been submitted successfully.",
                    from_=twilio_phone,
                    to=user_phone
                )
                logger.info(f"Confirmation SMS sent to user {user_phone} for report {report_id}")
            except Exception as e:
                logger.error(f"Failed to send confirmation SMS to user {user_phone}: {str(e)}")
        else:
            logger.warning("No user phone number provided, skipping confirmation SMS")
        # Send SMS to officer for valid reports
        if valid and officer_phone and category != "unknown":
            try:
                client_twilio.messages.create(
                    body=f"New {category} issue reported at {address}. Please investigate. Report ID: {report_id}",
                    from_=twilio_phone,
                    to=officer_phone
                )
                logger.info(f"SMS sent to {officer_phone} for report {report_id} at {address}")
            except Exception as e:
                logger.error(f"Failed to send SMS to {officer_phone}: {str(e)}")
        risk_score = 0.0
        if predictive_model and scaler:
            X = [[float(lat), float(lng)]]
            X_scaled = scaler.transform(X)
            risk_score = predictive_model.predict_proba(X_scaled)[0, 1]
            logger.info(f"Risk score calculated: {risk_score} for lat={lat}, lng={lng}")
            if risk_score > 0.5 and officer_phone and category != "unknown":
                try:
                    client_twilio.messages.create(
                        body=f"High-risk {category} issue reported at {address}. Risk score: {round(risk_score, 2)}. Report ID: {report_id}",
                        from_=twilio_phone,
                        to=officer_phone
                    )
                    logger.info(f"High-risk SMS sent to {officer_phone} for {address}")
                except Exception as e:
                    logger.error(f"Failed to send high-risk SMS to {officer_phone}: {str(e)}")
        else:
            logger.warning("Predictive model not available, risk_score set to 0.0")
        retrain_predictive_model()
        logger.info(f"Prediction: {category}, Confidence: {confidence}, Risk Score: {risk_score}")
        return jsonify({
            "prediction": category,
            "confidence": round(confidence, 2),
            "valid": valid,
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "assigned_officer": assigned_officer_name,
            "risk_score": round(float(risk_score), 2),
            "report_id": report_id
        })
    except Exception as e:
        logger.error(f"Error in predict_water_issue: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Predict water quality
@app.route("/predict_water_quality", methods=["POST"])
@user_token_required
def predict_water_quality(current_user):
    logger.info("Received predict_water_quality request")
    try:
        data = request.json
        lat = data.get("latitude")
        lng = data.get("longitude")
        ph = data.get("ph")
        turbidity = data.get("turbidity")
        temperature = data.get("temperature")
        conductivity = data.get("conductivity")
        simulation_id = data.get("simulation_id")
        if not all([lat, lng]):
            logger.error("Latitude and longitude are required")
            return jsonify({"error": "Latitude and longitude are required"}), 400
        try:
            lat = float(lat)
            lng = float(lng)
        except ValueError:
            logger.error("Invalid coordinate format")
            return jsonify({"error": "Coordinates must be numeric"}), 400
        try:
            response = requests.get(
                f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json",
                headers={"User-Agent": "WaterQualityApp/1.0"}
            )
            response.raise_for_status()
            address_data = response.json()
            address = address_data.get("display_name", "Unknown address")
        except Exception as e:
            logger.error(f"Failed to geocode coordinates: {str(e)}")
            address = "Unknown address"
        if simulation_id:
            sim_data = water_quality_collection.find_one({"simulation_id": simulation_id})
            if not sim_data:
                logger.error("Invalid simulation ID")
                return jsonify({"error": "Invalid simulation ID"}), 400
            ph = sim_data["ph"]
            turbidity = sim_data["turbidity"]
            temperature = sim_data["temperature"]
            conductivity = sim_data["conductivity"]
            logger.info(f"Using simulated IoT data: {sim_data}")
        else:
            manual_params = all([ph is not None, turbidity is not None, temperature is not None, conductivity is not None])
            if manual_params:
                try:
                    ph = float(ph)
                    turbidity = float(turbidity)
                    temperature = float(temperature)
                    conductivity = float(conductivity)
                except ValueError:
                    logger.error("Invalid parameter format")
                    return jsonify({"error": "Parameters must be numeric"}), 400
            else:
                recent_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
                nearby_predictions = list(water_quality_collection.find(
                    {
                        "created_at": {"$gte": recent_date},
                        "latitude": {"$exists": True},
                        "longitude": {"$exists": True}
                    },
                    {"ph": 1, "turbidity": 1, "temperature": 1, "conductivity": 1, "latitude": 1, "longitude": 1}
                ))
                nearby_params = []
                for pred in nearby_predictions:
                    distance = haversine_distance(lat, lng, pred["latitude"], pred["longitude"])
                    if distance <= 1:
                        nearby_params.append({
                            "ph": pred["ph"],
                            "turbidity": pred["turbidity"],
                            "temperature": pred["temperature"],
                            "conductivity": pred["conductivity"]
                        })
                if nearby_params:
                    ph = sum(p["ph"] for p in nearby_params) / len(nearby_params)
                    turbidity = sum(p["turbidity"] for p in nearby_params) / len(nearby_params)
                    temperature = sum(p["temperature"] for p in nearby_params) / len(nearby_params)
                    conductivity = sum(p["conductivity"] for p in nearby_params) / len(nearby_params)
                else:
                    ph = random.uniform(6.5, 8.5)
                    turbidity = random.uniform(0, 10)
                    temperature = random.uniform(15, 30)
                    conductivity = random.uniform(100, 1000)
        if not (0 <= ph <= 14 and 0 <= turbidity <= 100 and 0 <= temperature <= 100 and 0 <= conductivity <= 2000):
            logger.error("Parameters out of valid range")
            return jsonify({"error": "Parameters out of valid range"}), 400
        input_data = np.array([[ph, turbidity, temperature, conductivity]])
        prediction = water_quality_model.predict(input_data)[0]
        confidence = float(water_quality_model.predict_proba(input_data)[0][prediction])
        quality = "potable" if prediction == 1 else "contaminated"
        assigned_officer_name = "No available officer"
        officer_phone = None
        if quality == "contaminated":
            officer = assign_officer()
            if officer and "name" in officer:
                assigned_officer_name = officer["name"]
                officer_phone = officer.get("phone", "N/A")
                try:
                    client_twilio.messages.create(
                        body=f"Contaminated water detected at {address}. Please investigate.",
                        from_=twilio_phone,
                        to=officer_phone
                    )
                    logger.info(f"SMS sent to {officer_phone} for contaminated water at {address}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {officer_phone}: {str(e)}")
        prediction_id = str(uuid.uuid4())
        quality_data = {
            "prediction_id": prediction_id,
            "user_phone": current_user["phone"],
            "ph": ph,
            "turbidity": turbidity,
            "temperature": temperature,
            "conductivity": conductivity,
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "quality": quality,
            "confidence": round(confidence, 2),
            "assigned_officer": assigned_officer_name,
            "created_at": datetime.datetime.utcnow(),
            "simulation_id": simulation_id if simulation_id else None
        }
        water_quality_collection.insert_one(quality_data)
        logger.info(f"Water quality prediction: {quality}, Confidence: {confidence}")
        return jsonify({
            "prediction_id": prediction_id,
            "quality": quality,
            "confidence": round(confidence, 2),
            "parameters": {"ph": ph, "turbidity": turbidity, "temperature": temperature, "conductivity": conductivity},
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "assigned_officer": assigned_officer_name,
            "simulation_id": simulation_id if simulation_id else None
        })
    except Exception as e:
        logger.error(f"Error in predict_water_quality: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Community leaderboard
@app.route("/community_leaderboard", methods=["GET"])
def get_community_leaderboard():
    logger.info("Received get_community_leaderboard request")
    try:
        pipeline = [
            {
                "$group": {
                    "_id": "$user_phone",
                    "total_upvotes": {"$sum": {"$ifNull": ["$upvotes", 0]}},
                    "report_count": {"$sum": 1}
                }
            },
            {
                "$lookup": {
                    "from": "Users",
                    "localField": "_id",
                    "foreignField": "phone",
                    "as": "user"
                }
            },
            {
                "$unwind": {
                    "path": "$user",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {
                "$project": {
                    "user_phone": "$_id",
                    "user_name": {"$ifNull": ["$user.name", "Unknown User"]},
                    "total_upvotes": 1,
                    "report_count": 1,
                    "_id": 0
                }
            },
            {"$sort": {"total_upvotes": -1, "report_count": -1}},
            {"$limit": 10}
        ]
        leaderboard = list(water_reports_collection.aggregate(pipeline))
        logger.info(f"Returning {len(leaderboard)} leaderboard entries")
        return jsonify(leaderboard)
    except Exception as e:
        logger.error(f"Error fetching community leaderboard: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Water quality insights
@app.route("/water_quality_insights", methods=["POST"])
@user_token_required
def water_quality_insights(current_user):
    logger.info("Received water_quality_insights request")
    try:
        data = request.json
        prediction_id = data.get("prediction_id")
        if not prediction_id:
            logger.error("Prediction ID is required")
            return jsonify({"error": "Prediction ID is required"}), 400
        prediction = water_quality_collection.find_one({"prediction_id": prediction_id})
        if not prediction:
            logger.error("Prediction not found")
            return jsonify({"error": "Prediction not found"}), 404
        system_prompt = (
            "You are an expert in water quality analysis. Based on the provided water quality prediction data, "
            "generate a concise and actionable insight about the water quality at the specified location. "
            "Include recommendations for improvement if the water is contaminated. "
            "Keep the response under 100 words and use clear, professional language."
        )
        user_message = (
            f"Water quality prediction at {prediction['address']}:\n"
            f"- Quality: {prediction['quality']}\n"
            f"- Confidence: {prediction['confidence']}\n"
            f"- Parameters: pH={prediction['ph']}, Turbidity={prediction['turbidity']} NTU, "
            f"Temperature={prediction['temperature']}°C, Conductivity={prediction['conductivity']} µS/cm\n"
            f"Provide an actionable insight and recommendation."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=150
        )
        insight = chat_completion.choices[0].message.content
        logger.info(f"Generated insight: {insight}")
        return jsonify({"insight": insight})
    except Exception as e:
        logger.error(f"Error in water_quality_insights: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Upvote a report
@app.route("/upvote_report", methods=["POST"])
@user_token_required
def upvote_report(current_user):
    logger.info("Received upvote_report request")
    try:
        data = request.json
        report_id = data.get("report_id")
        if not report_id:
            logger.error("Report ID is required")
            return jsonify({"error": "Report ID is required"}), 400
        report = water_reports_collection.find_one({"_id": ObjectId(report_id)})
        if not report:
            logger.error("Report not found")
            return jsonify({"error": "Report not found"}), 404
        if current_user["phone"] in report.get("upvoted_by", []):
            logger.error("User has already upvoted this report")
            return jsonify({"error": "You have already upvoted this report"}), 400
        if "upvotes" not in report:
            water_reports_collection.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {"upvotes": 0, "upvoted_by": []}}
            )
        result = water_reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {
                "$inc": {"upvotes": 1},
                "$push": {"upvoted_by": current_user["phone"]}
            }
        )
        if result.modified_count > 0:
            logger.info(f"Report {report_id} upvoted by {current_user['phone']}")
            return jsonify({"message": "Report upvoted successfully"})
        else:
            logger.error("Failed to upvote report")
            return jsonify({"error": "Failed to upvote report"}), 500
    except Exception as e:
        logger.error(f"Error in upvote_report: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Get comments for a report
@app.route("/get_comments", methods=["GET"])
def get_comments():
    logger.info("Received get_comments request")
    try:
        report_id = request.args.get("report_id")
        if not report_id:
            logger.error("Report ID is required")
            return jsonify({"error": "Report ID is required"}), 400
        report = water_reports_collection.find_one({"_id": ObjectId(report_id)})
        if not report:
            logger.error("Report not found")
            return jsonify({"error": "Report not found"}), 404
        comments = list(comments_collection.find(
            {"report_id": report_id},
            {"_id": 1, "user_phone": 1, "user_name": 1, "comment": 1, "created_at": 1}
        ))
        for comment in comments:
            comment["_id"] = str(comment["_id"])
            if isinstance(comment["created_at"], datetime.datetime):
                comment["created_at"] = comment["created_at"].isoformat()
        logger.info(f"Returning {len(comments)} comments for report {report_id}")
        return jsonify(comments)
    except Exception as e:
        logger.error(f"Error fetching comments: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add a comment to a report
@app.route("/add_comment", methods=["POST"])
@user_token_required
def add_comment(current_user):
    logger.info("Received add_comment request")
    try:
        data = request.json
        report_id = data.get("report_id")
        comment_text = data.get("comment")
        if not report_id or not comment_text:
            logger.error("Report ID and comment are required")
            return jsonify({"error": "Report ID and comment are required"}), 400
        report = water_reports_collection.find_one({"_id": ObjectId(report_id)})
        if not report:
            logger.error("Report not found")
            return jsonify({"error": "Report not found"}), 404
        comment_data = {
            "report_id": report_id,
            "user_phone": current_user["phone"],
            "user_name": current_user["name"],
            "comment": comment_text,
            "created_at": datetime.datetime.utcnow()
        }
        comments_collection.insert_one(comment_data)
        logger.info(f"Comment added to report {report_id} by {current_user['phone']}")
        return jsonify({"message": "Comment added successfully"})
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        return jsonify({"error": str(e)}), 500

# User reports
@app.route("/user/reports", methods=["OPTIONS"])
def user_reports_options():
    response = jsonify({"message": "OK"})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Methods", "GET,OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,x-access-token")
    return response

# Community dashboard: Public reports for upvoting
@app.route("/community_reports", methods=["GET"])
def get_community_reports():
    logger.info("Received get_community_reports request")
    try:
        reports = list(water_reports_collection.find(
            {"resolved": False},
            {"_id": 1, "latitude": 1, "longitude": 1, "address": 1, "status": 1, "confidence": 1, "image": 1, "created_at": 1, "upvotes": 1}
        ))
        for report in reports:
            report["_id"] = str(report["_id"])
            if isinstance(report["created_at"], datetime.datetime):
                report["created_at"] = report["created_at"].isoformat()
        logger.info(f"Returning {len(reports)} community reports")
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error fetching community reports: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/user/reports", methods=["GET"])
@user_token_required
def get_user_reports(current_user):
    logger.info(f"Fetching reports for user phone: {current_user['phone']}")
    try:
        reports = list(water_reports_collection.find(
            {"user_phone": current_user["phone"]},
            {"_id": 1, "latitude": 1, "longitude": 1, "address": 1, "status": 1, "confidence": 1, "assigned_officer": 1, "officer_email": 1, "officer_phone": 1, "image": 1, "created_at": 1, "resolved": 1, "resolved_image": 1}
        ))
        for report in reports:
            report["_id"] = str(report["_id"])
            if isinstance(report["created_at"], datetime.datetime):
                report["created_at"] = report["created_at"].isoformat()
        logger.info(f"Found {len(reports)} reports")
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error fetching user reports: {str(e)}")
        return jsonify({"error": str(e)}), 500

# User authentication
@app.route("/user/send_otp", methods=["POST"])
def send_otp():
    logger.info("Received send_otp request")
    try:
        data = request.json
        phone = data.get("phone")
        if not phone:
            logger.error("Phone number is required")
            return jsonify({"error": "Phone number is required"}), 400
        if users_collection.find_one({"phone": phone}):
            logger.error("Phone number already registered")
            return jsonify({"error": "Phone number already registered"}), 400
        otp = str(random.randint(100000, 999999))
        otp_cache[phone] = otp
        try:
            client_twilio.messages.create(
                body=f"Your OTP for registration is: {otp}",
                from_=twilio_phone,
                to=phone
            )
            logger.info(f"OTP sent to {phone}")
            return jsonify({"message": "OTP sent successfully"})
        except Exception as e:
            logger.error(f"Failed to send OTP to {phone}: {str(e)}")
            return jsonify({"error": f"Failed to send OTP: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error in send_otp: {str(e)}")
        return jsonify({"error": f"Failed to send OTP: {str(e)}"}), 500

@app.route("/user/register", methods=["POST"])
def user_register():
    logger.info("Received user_register request")
    try:
        data = request.json
        name = data.get("name")
        phone = data.get("phone")
        email = data.get("email")
        address = data.get("address")
        aadhar = data.get("aadhar")
        password = data.get("password")
        otp = data.get("otp")
        if not all([name, phone, email, address, aadhar, password, otp]):
            logger.error("All fields are required")
            return jsonify({"error": "All fields are required"}), 400
        if otp_cache.get(phone) != otp:
            logger.error("Invalid or expired OTP")
            return jsonify({"error": "Invalid or expired OTP"}), 400
        hashed_password = generate_password_hash(password)
        user = {
            "name": name,
            "phone": phone,
            "email": email,
            "address": address,
            "aadhar": aadhar,
            "password": hashed_password,
            "created_at": datetime.datetime.utcnow()
        }
        users_collection.insert_one(user)
        del otp_cache[phone]
        logger.info(f"User registered: {email}")
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        logger.error(f"Error in user registration: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/user/login", methods=["POST"])
def user_login():
    logger.info("Received user_login request")
    try:
        data = request.json
        identifier = data.get("identifier")
        password = data.get("password")
        if not identifier or not password:
            logger.error("Email/Phone and password are required")
            return jsonify({"error": "Email/Phone and password are required"}), 400
        user = users_collection.find_one({"$or": [{"email": identifier}, {"phone": identifier}]})
        if not user or not check_password_hash(user["password"], password):
            logger.error("Invalid credentials")
            return jsonify({"error": "Invalid credentials"}), 401
        token = jwt.encode({
            "phone": user["phone"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        logger.info(f"User logged in: {identifier}")
        return jsonify({"token": token, "role": "user"})
    except Exception as e:
        logger.error(f"Error in user login: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Officer authentication
@app.route("/register", methods=["POST"])
def register():
    logger.info("Received officer register request")
    try:
        data = request.json
        if officers_collection.find_one({"email": data["email"]}):
            logger.error("Email already registered")
            return jsonify({"error": "Email already registered"}), 400
        hashed_password = generate_password_hash(data["password"])
        officer = {
            "name": data["name"],
            "email": data["email"],
            "phone": data["phone"],
            "password": hashed_password,
            "assigned_reports": 0
        }
        officers_collection.insert_one(officer)
        logger.info(f"Officer registered: {data['email']}")
        return jsonify({"message": "Registration successful"})
    except Exception as e:
        logger.error(f"Error in officer registration: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    logger.info("Received officer login request")
    try:
        data = request.json
        identifier = data.get("identifier")
        password = data.get("password")
        if not identifier or not password:
            logger.error("Email and password are required")
            return jsonify({"error": "Email and password are required"}), 400
        officer = officers_collection.find_one({"email": identifier})
        if not officer or not check_password_hash(officer["password"], password):
            logger.error("Invalid credentials")
            return jsonify({"error": "Invalid credentials"}), 401
        token = jwt.encode({
            "email": officer["email"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, app.config["SECRET_KEY"], algorithm="HS256")
        logger.info(f"Officer logged in: {identifier}")
        return jsonify({"token": token, "role": "officer"})
    except Exception as e:
        logger.error(f"Error in officer login: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Officer reports
@app.route("/officer/reports", methods=["GET"])
@token_required
def get_officer_reports(current_officer):
    logger.info("Received get_officer_reports request")
    try:
        reports = list(water_reports_collection.find(
            {"assigned_officer": current_officer["name"]},
            {
                "_id": 1,
                "address": 1,
                "status": 1,
                "assigned_officer": 1,
                "officer_email": 1,
                "officer_phone": 1,
                "image": 1,
                "confidence": 1,
                "resolved": 1,
                "progress": 1,
                "progress_notes": 1,
                "progress_image": 1
            }
        ))
        for report in reports:
            report["_id"] = str(report["_id"])
        logger.info(f"Returning {len(reports)} reports for officer {current_officer['name']}")
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error fetching officer reports: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/officer/resolved_reports", methods=["GET"])
@token_required
def get_officer_resolved_reports(current_officer):
    logger.info("Received get_officer_resolved_reports request")
    try:
        reports = list(water_reports_collection.find(
            {"assigned_officer": current_officer["name"], "resolved": True},
            {
                "_id": 1,
                "address": 1,
                "status": 1,
                "assigned_officer": 1,
                "officer_email": 1,
                "officer_phone": 1,
                "image": 1,
                "confidence": 1,
                "resolved": 1,
                "resolved_image": 1,
                "created_at": 1
            }
        ))
        for report in reports:
            report["_id"] = str(report["_id"])
            if isinstance(report["created_at"], datetime.datetime):
                report["created_at"] = report["created_at"].isoformat()
        logger.info(f"Returning {len(reports)} resolved reports for officer {current_officer['name']}")
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error fetching resolved reports: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Accept a report
@app.route("/accept_report", methods=["PUT"])
@token_required
def accept_report(current_officer):
    logger.info("Received accept_report request")
    try:
        data = request.json
        report_id = data.get("report_id")
        if not report_id:
            logger.error("Report ID is required")
            return jsonify({"error": "Report ID is required"}), 400
        report = water_reports_collection.find_one({
            "_id": ObjectId(report_id),
            "assigned_officer": current_officer["name"]
        })
        if not report:
            logger.error("Report not found or not assigned to officer")
            return jsonify({"error": "Report not found or not assigned to officer"}), 404
        result = water_reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {"status": "Accepted"}}
        )
        if result.modified_count > 0:
            send_notification(report_id, "Accepted")
            logger.info(f"Report {report_id} accepted by officer {current_officer['name']}")
            return jsonify({"message": "Report accepted successfully"})
        else:
            logger.error("Failed to accept report")
            return jsonify({"error": "Failed to accept report"}), 500
    except Exception as e:
        logger.error(f"Error in accept_report: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Update report progress
@app.route("/update_report_progress", methods=["PUT"])
@token_required
def update_report_progress(current_officer):
    logger.info("Received update_report_progress request")
    try:
        report_id = request.form.get("report_id")
        progress = request.form.get("progress")
        notes = request.form.get("notes", "")
        progress_image = request.files.get("progress_image")
        if not report_id:
            logger.error("Report ID is required")
            return jsonify({"error": "Report ID is required"}), 400
        report = water_reports_collection.find_one({
            "_id": ObjectId(report_id),
            "assigned_officer": current_officer["name"]
        })
        if not report:
            logger.error("Report not found or not assigned to officer")
            return jsonify({"error": "Report not found or not assigned to officer"}), 404
        update_data = {
            "status": "In-Progress",
            "progress": int(progress) if progress else report.get("progress", 0),
            "progress_notes": notes
        }
        if progress_image:
            progress_filename = f"{report_id}_progress.jpg"
            progress_path = os.path.join(app.config["UPLOAD_FOLDER"], progress_filename)
            progress_image.save(progress_path)
            update_data["progress_image"] = f"http://localhost:5000/uploads/{progress_filename}"
        result = water_reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            send_notification(report_id, f"In-Progress: {update_data['progress']}%")
            logger.info(f"Progress updated for report {report_id} by officer {current_officer['name']}")
            return jsonify({"message": "Progress updated successfully"})
        else:
            logger.error("Failed to update progress")
            return jsonify({"error": "Failed to update progress"}), 500
    except Exception as e:
        logger.error(f"Error in update_report_progress: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Update report status
@app.route("/update_report_status", methods=["PUT"])
@token_required
def update_report_status(current_officer):
    logger.info("Received update_report_status request")
    try:
        report_id = request.form.get("report_id")
        resolved_image = request.files.get("resolved_image")
        if not report_id:
            logger.error("Report ID is required")
            return jsonify({"error": "Report ID is required"}), 400
        report = water_reports_collection.find_one({
            "_id": ObjectId(report_id),
            "assigned_officer": current_officer["name"]
        })
        if not report:
            logger.error("Report not found or not assigned to officer")
            return jsonify({"error": "Report not found or not assigned to officer"}), 404
        update_data = {
            "resolved": True,
            "status": "Resolved"
        }
        if resolved_image:
            resolved_filename = f"{report_id}_resolved.jpg"
            resolved_path = os.path.join(app.config["RESOLVED_FOLDER"], resolved_filename)
            resolved_image.save(resolved_path)
            update_data["resolved_image"] = f"http://localhost:5000/resolved_images/{resolved_filename}"
        result = water_reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            send_notification(report_id, "Resolved")
            logger.info(f"Report {report_id} marked as resolved by officer {current_officer['name']}")
            return jsonify({"message": "Report marked as resolved"})
        else:
            logger.error("Failed to mark report as resolved")
            return jsonify({"error": "Failed to mark report as resolved"}), 500
    except Exception as e:
        logger.error(f"Error updating report status: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Profile endpoints
@app.route("/user/profile", methods=["GET"])
@user_token_required
def user_profile(current_user):
    logger.info("Received user_profile request")
    try:
        user_data = {
            "name": current_user["name"],
            "phone": current_user["phone"],
            "email": current_user["email"],
            "address": current_user["address"],
            "aadhar": current_user["aadhar"],
            "created_at": current_user["created_at"].isoformat() if isinstance(current_user["created_at"], datetime.datetime) else current_user["created_at"]
        }
        logger.info(f"Returning profile for user {current_user['phone']}")
        return jsonify(user_data)
    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/officer/profile", methods=["GET"])
@token_required
def officer_profile(current_officer):
    logger.info("Received officer_profile request")
    try:
        officer_data = {
            "name": current_officer["name"],
            "email": current_officer["email"],
            "phone": current_officer["phone"],
            "assigned_reports": current_officer["assigned_reports"]
        }
        resolved_count = water_reports_collection.count_documents({
            "assigned_officer": current_officer["name"],
            "resolved": True
        })
        total_assigned = officer_data["assigned_reports"]
        contribution = (resolved_count / total_assigned * 100) if total_assigned > 0 else 0
        officer_data["resolved_reports"] = resolved_count
        officer_data["contribution"] = round(contribution, 2)
        logger.info(f"Returning profile for officer {current_officer['name']}")
        return jsonify(officer_data)
    except Exception as e:
        logger.error(f"Error fetching officer profile: {str(e)}")
        return jsonify({"error": str(e)}), 500

# File serving
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        logger.error(f"Error serving uploaded file {filename}: {str(e)}")
        return jsonify({"error": "File not found"}), 404

@app.route('/resolved_images/<filename>')
def serve_resolved_image(filename):
    try:
        return send_from_directory(app.config["RESOLVED_FOLDER"], filename)
    except Exception as e:
        logger.error(f"Error serving resolved image {filename}: {str(e)}")
        return jsonify({"error": "File not found"}), 404

# Get all reports
@app.route("/get_reports", methods=["GET"])
def get_reports():
    logger.info("Received get_reports request")
    try:
        reports = list(water_reports_collection.find({}, {"_id": 0}))
        logger.info(f"Returning {len(reports)} reports")
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error fetching reports: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Simulate IoT data
@app.route("/simulate_iot_data", methods=["POST"])
@user_token_required
def simulate_iot_data(current_user):
    logger.info("Received simulate_iot_data request")
    try:
        data = request.json
        lat = data.get("latitude")
        lng = data.get("longitude")
        if not lat or not lng:
            logger.error("Latitude and longitude are required")
            return jsonify({"error": "Latitude and longitude are required"}), 400
        try:
            lat = float(lat)
            lng = float(lng)
        except ValueError:
            logger.error("Invalid coordinate format")
            return jsonify({"error": "Coordinates must be numeric"}), 400
        try:
            response = requests.get(
                f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json",
                headers={"User-Agent": "WaterQualityApp/1.0"}
            )
            response.raise_for_status()
            address_data = response.json()
            address = address_data.get("display_name", "Unknown address")
            logger.info(f"Geocoded address: {address} for lat={lat}, lng={lng}")
        except Exception as e:
            logger.error(f"Failed to geocode coordinates: {str(e)}")
            address = "Unknown address"
        sensor_data = generate_sensor_data(lat, lng)
        simulation_id = str(uuid.uuid4())
        simulation_data = {
            "simulation_id": simulation_id,
            "user_phone": current_user["phone"],
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "ph": sensor_data["ph"],
            "turbidity": sensor_data["turbidity"],
            "temperature": sensor_data["temperature"],
            "conductivity": sensor_data["conductivity"],
            "created_at": datetime.datetime.utcnow(),
            "source": "simulated_iot"
        }
        water_quality_collection.insert_one(simulation_data)
        logger.info(f"Simulated IoT data for {address}: {sensor_data}")
        return jsonify({
            "simulation_id": simulation_id,
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "sensor_data": sensor_data
        })
    except Exception as e:
        logger.error(f"Error in simulate_iot_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_sensor_data(lat, lng):
    try:
        recent_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        nearby_predictions = list(water_quality_collection.find(
            {
                "created_at": {"$gte": recent_date},
                "latitude": {"$exists": True},
                "longitude": {"$exists": True}
            },
            {"ph": 1, "turbidity": 1, "temperature": 1, "conductivity": 1, "latitude": 1, "longitude": 1}
        ))
        nearby_params = []
        for pred in nearby_predictions:
            distance = haversine_distance(lat, lng, pred["latitude"], pred["longitude"])
            if distance <= 1:
                nearby_params.append({
                    "ph": pred["ph"],
                    "turbidity": pred["turbidity"],
                    "temperature": pred["temperature"],
                    "conductivity": pred["conductivity"]
                })
        nearby_issues = list(water_reports_collection.find(
            {
                "created_at": {"$gte": recent_date},
                "latitude": {"$exists": True},
                "longitude": {"$exists": True}
            },
            {"status": 1, "latitude": 1, "longitude": 1}
        ))
        has_pollution = any(
            issue["status"] == "pollution" and
            haversine_distance(lat, lng, issue["latitude"], issue["longitude"]) <= 1
            for issue in nearby_issues
        )
        ph = random.uniform(6.5, 8.5)
        turbidity = random.uniform(0, 10)
        temperature = random.uniform(15, 30)
        conductivity = random.uniform(100, 1000)
        if nearby_params:
            avg_ph = sum(p["ph"] for p in nearby_params) / len(nearby_params)
            avg_turbidity = sum(p["turbidity"] for p in nearby_params) / len(nearby_params)
            avg_temperature = sum(p["temperature"] for p in nearby_params) / len(nearby_params)
            avg_conductivity = sum(p["conductivity"] for p in nearby_params) / len(nearby_params)
            ph = random.uniform(max(0, avg_ph - 0.5), min(14, avg_ph + 0.5))
            turbidity = random.uniform(max(0, avg_turbidity - 2), avg_turbidity + 2)
            temperature = random.uniform(max(0, avg_temperature - 5), avg_temperature + 5)
            conductivity = random.uniform(max(0, avg_conductivity - 100), avg_conductivity + 100)
        if has_pollution:
            ph = random.uniform(5.5, 7.0)
            turbidity = random.uniform(10, 50)
            conductivity = random.uniform(500, 1500)
        ph = max(0, min(14, ph))
        turbidity = max(0, min(100, turbidity))
        temperature = max(0, min(100, temperature))
        conductivity = max(0, min(2000, conductivity))
        return {
            "ph": round(ph, 2),
            "turbidity": round(turbidity, 2),
            "temperature": round(temperature, 2),
            "conductivity": round(conductivity, 2)
        }
    except Exception as e:
        logger.error(f"Error generating sensor data: {str(e)}")
        return {
            "ph": round(random.uniform(6.5, 8.5), 2),
            "turbidity": round(random.uniform(0, 10), 2),
            "temperature": round(random.uniform(15, 30), 2),
            "conductivity": round(random.uniform(100, 1000), 2)
        }

# Train clustering model for flow optimization
def train_clustering_model(reports):
    try:
        if len(reports) < 3:
            logger.warning("Insufficient reports for clustering")
            return None, None
        X = []
        for report in reports:
            severity = 0.8 if report["status"] == "scarcity" else 0.6 if report["status"] == "leakage" else 0.2
            X.append([report["latitude"], report["longitude"], severity])
        X = np.array(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=min(3, len(X)), random_state=42)
        kmeans.fit(X_scaled)
        logger.info("Clustering model trained successfully")
        return kmeans, scaler
    except Exception as e:
        logger.error(f"Error training clustering model: {str(e)}")
        return None, None

# Generate flow recommendations based on clusters
def generate_recommendations(kmeans, scaler, reports):
    try:
        X = [[r["latitude"], r["longitude"], 0.8 if r["status"] == "scarcity" else 0.6 if r["status"] == "leakage" else 0.2] for r in reports]
        X_scaled = scaler.transform(X)
        labels = kmeans.predict(X_scaled)
        recommendations = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_reports = [r for i, r in enumerate(reports) if labels[i] == cluster_id]
            if not cluster_reports:
                continue
            cluster_center = scaler.inverse_transform(kmeans.cluster_centers_)[cluster_id][:2]
            scarcity_count = sum(1 for r in cluster_reports if r["status"] == "scarcity")
            leakage_count = sum(1 for r in cluster_reports if r["status"] == "leakage")
            if scarcity_count > leakage_count:
                recommendation = f"Redirect {min(20, scarcity_count * 5)}% water flow to area near ({cluster_center[0]:.4f}, {cluster_center[1]:.4f}) due to high scarcity."
            else:
                recommendation = f"Repair leaks near ({cluster_center[0]:.4f}, {cluster_center[1]:.4f}) to reduce {min(20, leakage_count * 5)}% water loss."
            try:
                response = requests.get(
                    f"https://nominatim.openstreetmap.org/reverse?lat={cluster_center[0]}&lon={cluster_center[1]}&format=json",
                    headers={"User-Agent": "WaterQualityApp/1.0"}
                )
                response.raise_for_status()
                address = response.json().get("display_name", "Unknown address")
            except Exception as e:
                logger.error(f"Failed to geocode cluster center: {str(e)}")
                address = "Unknown address"
            recommendations.append({
                "cluster_id": int(cluster_id),
                "center": {"latitude": float(cluster_center[0]), "longitude": float(cluster_center[1])},
                "address": address,
                "scarcity_count": scarcity_count,
                "leakage_count": leakage_count,
                "recommendation": recommendation
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return []

# Optimize water flow
@app.route("/optimize_flow", methods=["POST"])
@token_required
def optimize_flow(current_officer):
    logger.info("Received optimize_flow request")
    try:
        recent_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        reports = list(water_reports_collection.find(
            {"created_at": {"$gte": recent_date}, "status": {"$in": ["scarcity", "leakage"]}},
            {"latitude": 1, "longitude": 1, "status": 1, "address": 1}
        ))
        if len(reports) < 3:
            logger.warning("Insufficient reports for flow optimization")
            return jsonify({"error": "Insufficient reports for optimization (need at least 3)"}), 400
        
        kmeans, scaler = train_clustering_model(reports)
        if not kmeans or not scaler:
            logger.error("Failed to train clustering model")
            return jsonify({"error": "Failed to train clustering model"}), 500
        
        recommendations = generate_recommendations(kmeans, scaler, reports)
        if not recommendations:
            logger.warning("No recommendations generated")
            return jsonify({"error": "No actionable recommendations generated"}), 400
        
        optimization_id = str(uuid.uuid4())
        optimization_data = {
            "optimization_id": optimization_id,
            "officer_email": current_officer["email"],
            "recommendations": recommendations,
            "created_at": datetime.datetime.utcnow()
        }
        flow_optimizations_collection.insert_one(optimization_data)

        sms_body = "Water flow optimization recommendations:\n" + "\n".join(
            [f"- {r['recommendation']} ({r['address']})" for r in recommendations]
        )
        try:
            client_twilio.messages.create(
                body=sms_body[:1600],
                from_=twilio_phone,
                to=current_officer["phone"]
            )
            logger.info(f"SMS sent to {current_officer['phone']} with optimization recommendations")
        except Exception as e:
            logger.error(f"Failed to send SMS to {current_officer['phone']}: {str(e)}")

        logger.info(f"Flow optimization completed: {optimization_id}")
        return jsonify({
            "optimization_id": optimization_id,
            "recommendations": recommendations
        })

    except Exception as e:
        logger.error(f"Error in optimize_flow: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Flow dashboard data
@app.route("/flow_dashboard", methods=["GET"])
@token_required
def flow_dashboard(current_officer):
    logger.info("Received flow_dashboard request")
    try:
        optimizations = list(flow_optimizations_collection.find(
            {"officer_email": current_officer["email"]},
            {"_id": 0, "optimization_id": 1, "recommendations": 1, "created_at": 1}
        ).sort("created_at", -1).limit(1))
        if not optimizations:
            logger.info("No optimization data available")
            return jsonify({"clusters": [], "recommendations": []})
        recent_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        reports = list(water_reports_collection.find(
            {"created_at": {"$gte": recent_date}, "status": {"$in": ["scarcity", "leakage"]}},
            {"latitude": 1, "longitude": 1, "status": 1, "address": 1}
        ))
        latest_optimization = optimizations[0]
        clusters = []
        for rec in latest_optimization["recommendations"]:
            clusters.append({
                "cluster_id": rec["cluster_id"],
                "center": rec["center"],
                "address": rec["address"],
                "scarcity_count": rec["scarcity_count"],
                "leakage_count": rec["leakage_count"],
                "recommendation": rec["recommendation"]
            })
        points = [
            {
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "status": r["status"],
                "address": r["address"]
            } for r in reports
        ]
        logger.info(f"Returning flow dashboard data with {len(clusters)} clusters")
        return jsonify({
            "clusters": clusters,
            "points": points,
            "optimization_id": latest_optimization["optimization_id"],
            "created_at": latest_optimization["created_at"].isoformat() if isinstance(latest_optimization["created_at"], datetime.datetime) else latest_optimization["created_at"]
        })
    except Exception as e:
        logger.error(f"Error in flow_dashboard: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    try:
        initialize_water_reports_schema()
        logger.info("Starting Flask application")
        app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}", exc_info=True)
