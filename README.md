# WaterWatchX

A Flask-based web application for detecting and reporting water issues (e.g., leakage, scarcity) using image uploads and location data. Features include SMS notifications, MongoDB storage, and officer assignment for issue resolution.

---

## Features
- Report water issues with images and location data.
- Categorize issues (leakage, scarcity, others).
- SMS notifications via Twilio.
- MongoDB for data storage.
- Pollution reports marked invalid with user verification.
- Officer assignment for issue resolution.

---

## Tech Stack
- **Backend**: Flask, Python
- **Database**: MongoDB
- **SMS**: Twilio API
- **Machine Learning**: CNN model for water issue detection (`water_cnn_model.h5`)

---

## Project Structure
- `app.py`: Main Flask application.
- `requirements.txt`: Python dependencies.
- `water_cnn_model.h5`: Pre-trained CNN model for water issue detection.
- `water_quality_model.pkl`: Model for water quality prediction.
- `test_model.py`: Script for testing the ML model.
- `train_model.py`: Script for training the ML model.
- `.gitignore`: Files and directories to ignore (e.g., `.env`, `uploads/`, `resolved_images/`).

**Note**: Image upload directories (`uploads/`, `resolved_images/`) are ignored in production and not included in the repository.

---

## Prerequisites
Before setting up the project, ensure you have the following installed:
- Python 3.8 or higher.
- MongoDB (local or cloud instance like MongoDB Atlas).
- A Twilio account for SMS notifications.
- (Optional) A GROQ API key if using GROQ for additional features.
- An SMTP server for email notifications (e.g., Gmail SMTP).

---

## Setup

Follow these steps to set up and run the WaterWatchX application:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HemanthGK2004/WaterWatchX.git
   cd WaterWatchX
2. **Set Up a Virtual Environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate
3. **Install Dependencies:**
   ```bash
      pip install -r requirements.txt
4. **Set up MongoDB:**
- Install MongoDB locally or use a cloud instance like MongoDB Atlas.
- Create a new database and collection for WaterWatchX data.
- Update the app.py file with your MongoDB connection details.
- Ensure MongoDB is running before proceeding.
- Note the MongoDB URI for later use.

5. **Configure Environment Variables**
   - Create a .env file in the root directory:
     ```bash
        # .env
      MONGODB_URI=mongodb://localhost:27017/waterwatchx  # Replace with your MongoDB URI
      GROQ_API_KEY=your_groq_api_key                     # Optional, if using GROQ
      TWILIO_SID=your_twilio_sid
      TWILIO_AUTH_TOKEN=your_twilio_auth_token
      TWILIO_PHONE=your_twilio_phone_number
      SECRET_KEY=your_flask_secret_key                   # A random string for Flask session security
      SMTP_HOST=smtp.gmail.com                           # Example: Gmail SMTP host
      SMTP_PORT=587                                      # Example: Gmail SMTP port
      SMTP_EMAIL=your_email@gmail.com                    # Your SMTP email
      SMTP_PASSWORD=your_smtp_password                   # Your SMTP password or app-specific password
6. **Run the Application**
   ```bash
      python app.py
**Machine Learning Models**
This project uses two machine learning models:
1. water_cnn_model.h5
A pre-trained Convolutional Neural Network (CNN) model for detecting water issues in images.
Capable of categorizing issues such as leakage, scarcity, or others.
2. water_quality_model.pkl
A model for predicting water quality based on input data.
Helps in determining if water is safe for use or requires attention.

**Usage**
   **Reporting a Water Issue**
      - Users can upload an image and provide location data via the web interface.
      - The system categorizes the issue as leakage, scarcity, or others.
   **Issue Verification**
      - Pollution reports are marked invalid and require manual user verification.
   **Notifications**
      - Receive SMS updates via Twilio for issue status changes or resolutions.
   **Officer Assignment**
      - Issues are automatically assigned to officers for resolution based on categorization and priority.


