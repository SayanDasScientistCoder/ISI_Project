# pages/ResetPassword.py
import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timezone
import bcrypt, hashlib
from config import MONGO_URI


st.set_page_config(page_title="Reset Password", layout="centered")

# ------------------ HIDE SIDEBAR ------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD CSS ------------------
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("styles/theme.css")

# ------------------ MONGO ------------------
#MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment variables")

client = MongoClient(MONGO_URI)
db = client["auth_db"]
users_collection = db["users"]
resets_collection = db["password_resets"]

# ------------------ TOP BAR ------------------
left, right = st.columns([6, 1])
with left:
    st.markdown("## Reset Password")
st.markdown("---")

# ------------------ READ TOKEN FROM SESSION ------------------
# Token should already be in session state from app.py
token_plain = st.session_state.get("incoming_reset_token", None)

if not token_plain:
    st.error("Invalid or expired link.")
    if st.button("Return to Login"):
        st.switch_page("pages/0_Login.py")
    st.stop()

# Hash the token for database lookup
token_hash = hashlib.sha256(token_plain.encode("utf-8")).hexdigest()
doc = resets_collection.find_one({"token_hash": token_hash})

if not doc:
    st.error("Invalid or expired reset token.")
    if st.button("Return to Login"):
        st.switch_page("pages/0_Login.py")
    st.stop()

# FIX: Handle both naive and aware datetime comparisons
expires_at = doc["expires_at"]
now = datetime.now(timezone.utc)

# If expires_at is naive (no timezone), assume it's UTC
if expires_at.tzinfo is None:
    expires_at = expires_at.replace(tzinfo=timezone.utc)

if now > expires_at:
    resets_collection.delete_one({"_id": doc["_id"]})
    st.error("This reset link has expired.")
    if st.button("Return to Login"):
        st.switch_page("pages/0_Login.py")
    st.stop()

email = doc["email"]

# ------------------ RESET FORM ------------------
with st.form("reset_pw_form"):
    new_pw = st.text_input("New Password", type="password")
    confirm_pw = st.text_input("Confirm Password", type="password")
    submit = st.form_submit_button("Update Password")

if submit:
    if len(new_pw) < 6:
        st.error("Password must be at least 6 characters.")
    elif new_pw != confirm_pw:
        st.error("Passwords do not match.")
    else:
        hashed_pw = bcrypt.hashpw(new_pw.encode("utf-8"), bcrypt.gensalt())
        users_collection.update_one({"email": email}, {"$set": {"password": hashed_pw}})

        resets_collection.delete_many({"email": email})
        st.session_state["incoming_reset_token"] = None

        st.success("Password updated successfully!")

        if st.button("Go to Login"):
            st.switch_page("pages/0_Login.py")