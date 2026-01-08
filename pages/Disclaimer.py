import streamlit as st
from pymongo import MongoClient
from config import MONGO_URI

st.set_page_config(
    page_title="Disclaimer",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        .css-1d391kg {display: none;}
    </style>
""", unsafe_allow_html=True)

# ------------------ DB CONNECTION ------------------
client = MongoClient(MONGO_URI)
db = client["auth_db"]
users_collection = db["users"]

# ------------------ AUTH GUARD ------------------
if not st.session_state.get("logged_in", False):
    st.switch_page("pages/0_Login.py")
    st.stop()

email = st.session_state.user

user = users_collection.find_one({"email": email})
if not user:
    st.error("User not found.")
    st.stop()

# If already accepted, skip this page
if user.get("tnc_accepted", False):
    st.switch_page("pages/3_Dashboard.py")
    st.stop()

# ------------------ DISCLAIMER CONTENT ------------------
st.markdown("## Disclaimer & Terms of Use")

st.info("""
-This software is an experimental tool developed for research and academic purposes in the field of computer vision and medical image analysis.

-Non-Diagnostic: This tool is NOT a medical device and has not been cleared by the FDA, EMA, or any other regulatory body. It must not be used for clinical diagnosis, staging, or treatment planning.

-No Liability: The results (segmentation masks, measurements, or classifications) are provided "as-is" without any warranty. The developers assume no responsibility for any medical decisions made based on this output.

-Consult a Professional: If you have concerns about a skin lesion, please consult a board-certified dermatologist immediately.
""")

st.markdown("---")

if st.button("I understand and agree to these terms", type="primary"):
    users_collection.update_one(
        {"email": email},
        {"$set": {"tnc_accepted": True}}
    )
    st.success("Thank you! Redirecting...")
    st.switch_page("pages/3_Dashboard.py")
