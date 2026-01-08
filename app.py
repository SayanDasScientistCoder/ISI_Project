import streamlit as st
from config import APP_NAME,MONGO_URI
from pymongo import MongoClient

st.set_page_config(
    page_title=APP_NAME,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ RESET TOKEN HANDLING ------------------
query_params = st.query_params
if "reset_token" in query_params:
    st.session_state["incoming_reset_token"] = query_params["reset_token"]
    st.switch_page("pages/ResetPassword.py")
    st.stop()

# ------------------ AUTH GATE ------------------
client = MongoClient(MONGO_URI)
db = client["auth_db"]
users_collection = db["users"]

if not st.session_state.get("logged_in", False):
    st.switch_page("pages/0_Login.py")
else:
    user = users_collection.find_one({"email": st.session_state.user})
    
    if not user.get("tnc_accepted", False):
        st.switch_page("pages/Disclaimer.py")
    else:
        st.switch_page("pages/3_Dashboard.py")
