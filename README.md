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
2.Set Up a Virtual Environment (optional but recommended):
```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows

3.**Install Dependencies:**:
```bash
   pip install -r requirements.txt


