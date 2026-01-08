import streamlit as st
import bcrypt, hashlib, secrets, smtplib, ssl, os
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from config import (
    MONGO_URI,
    GMAIL_ADDRESS,
    GMAIL_APP_PASSWORD,
    PASSWORD_RESET_EXPIRY_MINUTES
)

# ------------------ STREAMLIT PAGE CONFIG ------------------
st.set_page_config(page_title="Login", layout="centered")

# ------------------ HIDE SIDEBAR ------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD THEME CSS ------------------
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("styles/theme.css")

# ------------------ SESSION INIT ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Login"

# ------------------ MONGO SETUP ------------------
#MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment variables")

client = MongoClient(MONGO_URI)
db = client["auth_db"]
users_collection = db["users"]
resets_collection = db["password_resets"]

# ------------------ SECURITY HELPERS ------------------
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed)

def make_reset_token():
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=PASSWORD_RESET_EXPIRY_MINUTES
    )   
    return token, token_hash, expires_at

def send_reset_email(to_email: str, token_plain: str):
    link = f"http://localhost:8501/?reset_token={token_plain}"

    msg = EmailMessage()
    msg["Subject"] = "Password Reset Link"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email
    msg.set_content(
        f"Click below to reset your password:\n{link}\n\n"
        f"This link expires in {PASSWORD_RESET_EXPIRY_MINUTES} minutes."
    )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)

# ------------------ TOP BAR ------------------
left, right = st.columns([6, 1])
with left:
    st.markdown("## Login / Register")
st.markdown("---")

# ------------------ TABS ------------------
tabs = ["Login", "Register"]
selected = st.radio("", tabs, horizontal=True,
                    index=tabs.index(st.session_state.active_tab))
st.session_state.active_tab = selected

# ------------------ LOGIN TAB ------------------
if selected == "Login":
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_login = st.form_submit_button("Login")
    
    if submit_login:
        user = users_collection.find_one({"email": email})

        if user and check_password(password, user["password"]):
            users_collection.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.now(timezone.utc)}}
            )

            st.session_state.logged_in = True
            st.session_state.user = email
            user = users_collection.find_one({"email": email})

            if not user.get("tnc_accepted", False):
                st.switch_page("pages/Disclaimer.py")
            else:
                st.switch_page("pages/3_Dashboard.py")
        else:
            st.error("Invalid email or password")

    # Forgotten password UI
    with st.expander("Forgot your password?"):
        fp_email = st.text_input("Registered Email", key="fp_email")

        if st.button("Send Reset Link"):
            if fp_email:
                user = users_collection.find_one({"email": fp_email})

                if user:
                    token_plain, token_hash, expires_at = make_reset_token()

                    resets_collection.delete_many({"email": fp_email})
                    resets_collection.insert_one({
                        "email": fp_email,
                        "token_hash": token_hash,
                        "expires_at": expires_at
                    })

                    try:
                        send_reset_email(fp_email, token_plain)
                        st.success("Reset link sent! Check your inbox.")
                    except Exception as e:
                        st.error(f"Failed to send email: {e}")
                else:
                    # Do not reveal user existence
                    st.success("Reset link sent! Check your inbox.")

# ------------------ REGISTER TAB ------------------
elif selected == "Register":
    with st.form("register_form"):
        reg_email = st.text_input("Register Email")
        reg_password = st.text_input("Password", type="password")
        submit_reg = st.form_submit_button("Create Account")
    
    if submit_reg:
        if len(reg_password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            if users_collection.find_one({"email": reg_email}):
                st.error("User already exists!")
            else:
                hashed = hash_password(reg_password)
                users_collection.insert_one({
                    "email": reg_email,
                    "password": hashed,
                    "created_at": datetime.now(timezone.utc),
                    "last_login": None,
                    "total_uploads": 0,
                    "total_predictions": 0,
                    "account_type": "Free",
                    "tnc_accepted": False   
                })

                st.success("Account created! You may now log in.")