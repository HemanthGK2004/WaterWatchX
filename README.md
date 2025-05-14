# WaterWatchX 

A Flask-based web application for detecting and reporting water issues (leakage, scarcity, etc.) using image uploads and location data. Features SMS notifications, MongoDB storage, and officer assignment.

## Features
- Report water issues with images and location
- Categorize issues (leakage, scarcity, others)
- SMS notifications via Twilio
- MongoDB for data storage
- Pollution reports marked invalid with user verification
- Officer assignment for issue resolution

## Tech Stack
- Backend: Flask, Python
- Database: MongoDB
- SMS: Twilio API
- ML: CNN model for water issue detection (`water_cnn_model.h5`)
- Frontend: (Optional) React app in `healthcare-ai/` for visualization

## Project Structure
- `flask_backend/`: Flask backend for WaterWatchX
  - `app.py`: Main Flask application
  - `requirements.txt`: Python dependencies
  - `uploads/`: Directory for user-uploaded images (ignored in production)
  - `resolved_images/`: Directory for resolved issue images (ignored in production)
  - `water_cnn_model.h5`: Pre-trained CNN model for water issue detection
- `healthcare-ai/`: React frontend (optional, may be removed if unrelated)

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/HemanthGK2004/WaterWatchX.git
   cd WaterWatchX
   ```bash
2. Set up a virtual environment (optional but recommended):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```bash
3. Install MongoDB and Twilio API keys (see `app.py` for setup instructions)
4.create .env file with :
```bash
GROQ_API_KEY=
TWILIO_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE=
SECRET_KEY=
SMTP_HOST=
SMTP_PORT=
SMTP_EMAIL=
SMTP_PASSWORD=
5. Run the Flask app:
```bash
python app.py

