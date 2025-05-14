from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["WaterIssuesDB"]
collection = db["WaterReports"]

test_reports = [
    {
        "user_phone": "+919686353305",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "address": "Bangalore, Karnataka, India",
        "status": "scarcity",
        "confidence": 0.85,
        "assigned_officer": "Officer1",
        "officer_email": "officer1@example.com",
        "officer_phone": "+1234567891",
        "image": "http://localhost:5000/uploads/681f221915e1a112f08fc066_water_issue.jpg",
        "created_at": datetime.utcnow(),
        "resolved": False,
        "upvotes": 0,
        "upvoted_by": [],
        "progress": 0,
        "progress_notes": "",
        "progress_image": None
    },
    {
        "user_phone": "+919686353305",
        "latitude": 12.9616,
        "longitude": 77.5846,
        "address": "Bangalore, Karnataka, India",
        "status": "leakage",
        "confidence": 0.75,
        "assigned_officer": "Officer1",
        "officer_email": "officer1@example.com",
        "officer_phone": "+1234567891",
        "image": "http://localhost:5000/uploads/681ee49dcd45e9431df718d6_water_issue.jpg",
        "created_at": datetime.utcnow(),
        "resolved": False,
        "upvotes": 0,
        "upvoted_by": [],
        "progress": 0,
        "progress_notes": "",
        "progress_image": None
    }
]

collection.insert_many(test_reports)
print("Test reports inserted")