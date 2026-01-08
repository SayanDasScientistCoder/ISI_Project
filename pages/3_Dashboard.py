# pages/3_Dashboard.py

import streamlit as st
from datetime import datetime
from pymongo import MongoClient
from config import MONGO_URI

# ------------------ AUTH GUARD ------------------
if not st.session_state.get("logged_in", False):
    st.switch_page("pages/0_Login.py")
    st.stop()

# ------------------ DB CONNECTION ------------------
client = MongoClient(MONGO_URI)
db = client["auth_db"]
users_collection = db["users"]

user = users_collection.find_one({"email": st.session_state.user})
if not user:
    st.error("User record not found in database.")
    st.stop()

if not user.get("tnc_accepted", False):
    st.switch_page("pages/Disclaimer.py")
    st.stop()

# ------------------ EXTRACT REAL USER DATA ------------------
user_email = user["email"]
account_type = user.get("account_type", "Free")

created_at = user.get("created_at")
last_login = user.get("last_login")

total_uploads = user.get("total_uploads", 0)
total_predictions = user.get("total_predictions", 0)

def fmt(dt):
    return dt.strftime("%d %b %Y • %H:%M") if dt else "—"

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="User Dashboard", layout="centered")

# ------------------ HIDE SIDEBAR ------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD THEME ------------------
def load_css(path):
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css("styles/theme.css")

# ------------------ DASHBOARD STYLES ------------------
st.markdown("""
<style>
.dashboard-card {
    background: rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    transition: all 0.25s ease;
}
.dashboard-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 40px rgba(102,252,241,0.25);
}

.kpi-value {
    font-size: 36px;
    font-weight: 800;
    margin-top: 8px;
    color: #66fcf1;
}

.kpi-label {
    font-size: 15px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
h1, h2 = st.columns([6, 1])

with h1:
    st.markdown("## 👤 Dashboard")
    st.markdown(
        f"<span style='opacity:0.7'>Welcome back, <b>{user_email}</b></span>",
        unsafe_allow_html=True
    )

with h2:
    if st.button("🔄 Logout"):
        st.session_state.clear()
        st.switch_page("pages/0_Login.py")

st.markdown("---")

# ------------------ ACCOUNT OVERVIEW ------------------
st.markdown("### Account Overview")

st.markdown(f"""
<div class="dashboard-card">
    <p><b>Email:</b> {user_email}</p>
    <p><b>Plan:</b> {account_type}</p>
    <p><b>Member Since:</b> {fmt(created_at)}</p>
    <p><b>Last Active:</b> {fmt(last_login)}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------ KPI STATS ------------------
st.markdown("### Usage Summary")

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="kpi-label">📤 Total Uploads</div>
        <div class="kpi-value">{total_uploads}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="kpi-label">🧠 Predictions Run</div>
        <div class="kpi-value">{total_predictions}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="kpi-label">⭐ Subscription</div>
        <div class="kpi-value">{account_type}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------ ACTION CENTER ------------------
st.markdown("### Quick Actions")

a1, a2 = st.columns(2)

with a1:
    st.markdown("""
    <div class="dashboard-card">
        <h4>📤 Upload Images</h4>
        <p>Start a new prediction workflow</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Upload", use_container_width=True):
        st.switch_page("pages/1_Upload.py")

with a2:
    st.markdown("""
    <div class="dashboard-card">
        <h4>📊 View Results</h4>
        <p>Inspect predictions & reports</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Results", use_container_width=True):
        st.switch_page("pages/2_Result.py")

st.markdown("---")