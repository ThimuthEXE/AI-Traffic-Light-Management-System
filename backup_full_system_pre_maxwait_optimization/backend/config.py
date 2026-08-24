"""
Backend Configuration Module
Settings for FastAPI REST API, WebSockets, CORS, and MongoDB / Local Storage.
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Server Config
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# Database Config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "ai_traffic_db"
FALLBACK_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "db_fallback")

# CORS Origins for React Frontend
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"
]
