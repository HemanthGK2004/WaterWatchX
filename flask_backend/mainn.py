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
import jwt
import datetime
from functools import wraps
from bson import ObjectId
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")  # Load from .env or use default
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
    client.server_info()  # Test connection
    db = client["WaterIssuesDB"]
    water_reports_collection = db["WaterReports"]
    users_collection = db["Users"]
    officers_collection = db["Officers"]
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

otp_cache = {}

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
        category = ['leakage', 'pollution', 'scarcity'][max_idx]
        confidence = float(preds[max_idx])
        valid = confidence >= 0.6

        assigned_officer_name = "No available officer"
        officer_email = None
        officer_phone = None

        if valid:
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
                "resolved": False
            }
            water_reports_collection.insert_one(report_data)

        logger.info(f"Prediction: {category}, Confidence: {confidence}")
        return jsonify({
            "prediction": category,
            "confidence": round(confidence, 2),
            "valid": valid,
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "assigned_officer": assigned_officer_name
        })

    except Exception as e:
        logger.error(f"Error in predict_water_issue: {str(e)}")
        return jsonify({"error": str(e)}), 500

# User reports
@app.route("/user/reports", methods=["OPTIONS"])
def user_reports_options():
    response = jsonify({"message": "OK"})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Methods", "GET,OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,x-access-token")
    return response

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

        client_twilio.messages.create(
            body=f"Your OTP for registration is: {otp}",
            from_=twilio_phone,
            to=phone
        )
        logger.info(f"OTP sent to {phone}")
        return jsonify({"message": "OTP sent successfully"})
    except Exception as e:
        logger.error(f"Failed to send OTP: {str(e)}")
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
                "resolved": 1
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
            if "created_at" in report and isinstance(report["created_at"], datetime.datetime):
                report["created_at"] = report["created_at"].isoformat()

        logger.info(f"Returning {len(reports)} resolved reports for officer {current_officer['name']}")
        return jsonify(reports)
    except Exception as e:
        logger.error(f"Error fetching resolved reports: {str(e)}")
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

        update_data = {"resolved": True}
        if resolved_image:
            resolved_filename = f"{report_id}_resolved.jpg"
            resolved_path = os.path.join(app.config["RESOLVED_FOLDER"], resolved_filename)
            resolved_image.save(resolved_path)
            update_data["resolved_image"] = f"http://localhost:5000/resolved_images/{resolved_filename}"

        result = water_reports_collection.update_one(
            {"_id": ObjectId(report_id), "assigned_officer": current_officer["name"]},
            {"$set": update_data}
        )
        if result.modified_count > 0:
            logger.info(f"Report {report_id} marked as resolved")
            return jsonify({"message": "Report marked as resolved"})
        else:
            logger.error("Report not found or not assigned to officer")
            return jsonify({"error": "Report not found or not assigned to officer"}), 404
    except Exception as e:
        logger.error(f"Error updating report status: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    try:
        logger.info("Starting Flask application")
        app.run(debug=False, host="0.0.0.0", port=5000)  # Debug=False to prevent reloading; change to True for development
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}", exc_info=True)