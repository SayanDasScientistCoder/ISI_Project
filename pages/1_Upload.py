import streamlit as st
from PIL import Image
from pymongo import MongoClient
from config import MONGO_URI

if not st.session_state.get("logged_in", False):
    st.switch_page("pages/0_Login.py")
    st.stop()

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

# ---------------- Page + theme ----------------
st.set_page_config(layout="centered")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
document.addEventListener("DOMContentLoaded", function(){
    for (let i = 0; i < 25; i++) {
        let p = document.createElement("div");
        p.className = "particle";
        p.style.left = Math.random() * window.innerWidth + "px";
        p.style.top  = Math.random() * window.innerHeight + "px";
        p.style.animationDelay = (Math.random() * 10) + "s";
        document.body.appendChild(p);
    }
});
</script>
""", unsafe_allow_html=True)


def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def remove_image(idx_to_remove):
    """Callback function to remove an image"""
    ss["input_images"].pop(idx_to_remove)
    ss["input_filenames"].pop(idx_to_remove)
    
    st.rerun()


load_css("styles/theme.css")



# ---------------- Session init ----------------
ss = st.session_state
ss.setdefault("input_images", [])
ss.setdefault("input_filenames", [])
# used to prevent re-adding files from the uploader in the rerun right after a removal
ss.setdefault("_skip_add_once", False)

# ---------------- Header ----------------
c1, c2 = st.columns([6, 1])
with c1:
    st.markdown("## Upload Images")
with c2:
    if st.button("Logout"):
        st.session_state.clear()
        st.switch_page("app.py")

# ---------------- Top actions ----------------
a1, a2, a3 = st.columns(3)
with a1:
    if st.button("⬅️ Go Back"):
        st.switch_page("app.py")
with a2:
    if st.button("Predict ➜"):
        if not ss["input_images"]:
            st.warning("Please upload at least one image.")
        else:
            st.switch_page("pages/2_Result.py")
with a3:
    if st.button("🔄 Reset"):
        ss["input_images"] = []
        ss["input_filenames"] = []
        ss["_skip_add_once"] = False
        st.rerun()

st.markdown("---")

# ---------------- Uploader ----------------
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploader"
)

# Add from uploader only if we're NOT in the "post-removal rerun"
if uploaded_files and not ss["_skip_add_once"]:
    for file in uploaded_files:
        # Only add new names not already in the list
        if file.name not in ss["input_filenames"]:
            img = Image.open(file).convert("RGB")
            ss["input_images"].append(img)
            ss["input_filenames"].append(file.name)
    
#ss["_skip_add_once"] = False

# ---------------- Gallery with Remove buttons ----------------
if ss["input_images"]:
    st.subheader("Uploaded Images")
    
    # Use a dictionary to track removal selections
    if "removal_selections" not in ss:
        ss["removal_selections"] = {}
    
    cols = st.columns(3, gap="large")
    
    # Display images with checkboxes
    for idx, (img, fname) in enumerate(zip(ss["input_images"], ss["input_filenames"])):
        with cols[idx % 3]:
            st.image(img, width=220, caption=fname)
            # Use filename as key for checkbox
            ss["removal_selections"][fname] = st.checkbox(
                f"Remove {fname}", 
                key=f"remove_{fname}",
                value=ss["removal_selections"].get(fname, False)
            )
    
    st.markdown("---")
    
    # Count selected images
    selected_count = sum(1 for selected in ss["removal_selections"].values() if selected)
    
    if selected_count > 0:
        st.write(f"**{selected_count} image(s) selected for removal**")
    
    # Apply removal button
    if st.button(f"Remove {selected_count} Selected Image(s)", 
                 type="primary", 
                 disabled=selected_count == 0):
        
        # Mark that on the next rerun we should NOT add from the uploader
        ss["_skip_add_once"] = True
        
        # Create new lists for items to keep
        keep_imgs = []
        keep_names = []
        
        #for img, fname in zip(ss["input_images"], ss["input_filenames"]):
        #    if not ss["removal_selections"].get(fname, False):
        #        keep_imgs.append(img)
        #        keep_names.append(fname)

        for img, fname in zip(ss["input_images"], ss["input_filenames"]):
            if ss["removal_selections"][fname]==False:
                keep_imgs.append(img)
                keep_names.append(fname)

        # Update the session state
        ss["input_images"] = keep_imgs
        ss["input_filenames"] = keep_names
        
        # Clear removal selections for removed images
        # Only keep selections for images that still exist
        new_selections = {}
        for fname in keep_names:
            if fname in ss["removal_selections"]:
                new_selections[fname] = ss["removal_selections"][fname]
        ss["removal_selections"] = new_selections
        
        #if len(ss["input_filenames"])==0:
            #print("Type is",type(uploaded_files))
            #ss["_skip_add_once"]=False
        st.rerun()
    
    # Clear all selections button
    if st.button("Clear All Selections"):
        ss["removal_selections"] = {}
        st.rerun()

