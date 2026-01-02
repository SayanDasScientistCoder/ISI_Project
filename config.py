# config.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# =========================
# EMAIL CONFIG
# =========================
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
    raise RuntimeError("Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD in .env")

# =========================
# DATABASE CONFIG
# =========================
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI in .env")

# =========================
# APP CONFIG
# =========================
APP_NAME = "ML GUI"
PASSWORD_RESET_EXPIRY_MINUTES = 30
