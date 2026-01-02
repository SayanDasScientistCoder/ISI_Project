import streamlit as st
from config import APP_NAME

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
if not st.session_state.get("logged_in", False):
    st.switch_page("pages/0_Login.py")
else:
    st.switch_page("pages/3_Dashboard.py")
