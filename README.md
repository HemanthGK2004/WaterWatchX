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