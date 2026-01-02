# ============================================================
#                  2_Result.py  (ENSEMBLE + METRICS)
# ============================================================

import os
import io
import math
import zipfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms

import cv2
import streamlit as st

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

if not st.session_state.get("logged_in", False):
    st.switch_page("pages/0_Login.py")
    st.stop()

# --------------------- PAGE / SIDEBAR ---------------------
st.set_page_config(layout="centered")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        .css-1d391kg {display: none;}
    </style>
""", unsafe_allow_html=True)

# --------------------- LOAD CSS ---------------------
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("styles/theme.css")


# --------------------- HEADER ---------------------
header_left, header_right = st.columns([6, 1])
with header_left:
    st.markdown("## Prediction Results")
st.markdown("---")

if "input_images" not in st.session_state or not st.session_state["input_images"]:
    st.warning("Please upload images first.")
    st.stop()

# ============================================================
#                MODEL DISCOVERY + LOADING
# ============================================================

MODEL_DIR = "models"

def list_all_models():
    """List all .pt/.pth files in models/ directory."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith((".pt", ".pth"))]

def load_single_model(path, device="cpu"):
    """Load TorchScript (.pt) or torch.save (.pth) model on specified device."""
    full = os.path.join(MODEL_DIR, path)
    try:
        model = torch.jit.load(full, map_location=device)
    except Exception:
        model = torch.load(full, map_location=device)
    model.eval()
    return model

# ---- Model selector (top) ----
st.markdown("### 🧠 Select Models for Weighted Ensembling")

all_models = list_all_models()
if not all_models:
    st.error("❌ No models found in the `models/` folder.")
    st.stop()

selected_models = st.multiselect(
    "Choose one or more models:",
    options=all_models,
    default=[all_models[0]],
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# ---- Model Configuration Section ----
st.markdown("### ⚙️ Configure Each Model")

model_configs = {}
for m in selected_models:
    st.markdown(f"#### Configuration for `{m}`")
    
    # Input size configuration
    col1, col2 = st.columns(2)
    with col1:
        input_width = st.number_input(
            f"Input Width for {m}",
            min_value=64,
            max_value=1024,
            value=512 if "512" in m else 256,  # Default based on filename hint
            step=64,
            key=f"width_{m}"
        )
    with col2:
        input_height = st.number_input(
            f"Input Height for {m}",
            min_value=64,
            max_value=1024,
            value=512 if "512" in m else 256,
            step=64,
            key=f"height_{m}"
        )
    
    # Device selection
    device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    selected_device = st.selectbox(
        f"Device for {m}",
        options=device_options,
        index=0 if not torch.cuda.is_available() else 1,
        key=f"device_{m}"
    )
    
    # Weight assignment
    weight = st.slider(
        f"Weight for {m}",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        key=f"weight_{m}"
    )
    
    model_configs[m] = {
        "input_size": (int(input_height), int(input_width)),
        "device": selected_device,
        "weight": weight
    }

# Normalize weights
w_sum = sum(config["weight"] for config in model_configs.values())
if w_sum == 0:
    st.error("All model weights are zero. Increase at least one weight.")
    st.stop()

for m in model_configs:
    model_configs[m]["weight"] /= w_sum

# Load models with their specified devices
loaded_models = {}
for m in selected_models:
    try:
        device = model_configs[m]["device"]
        if device == "cuda" and not torch.cuda.is_available():
            st.warning(f"CUDA not available for {m}. Using CPU instead.")
            device = "cpu"
            model_configs[m]["device"] = "cpu"
        
        loaded_models[m] = load_single_model(m, device=device)
        st.success(f"✓ Loaded {m} on {device}")
    except Exception as e:
        st.error(f"❌ Failed to load {m}: {str(e)}")
        st.stop()

# ============================================================
#                    SESSION INPUT CHECK
# ============================================================

input_images = st.session_state["input_images"]
input_filenames = st.session_state["input_filenames"]

# Ensure download_selection exists AND matches current filenames
if "download_selection" not in st.session_state:
    st.session_state["download_selection"] = {}

# Add missing filenames
for fname in input_filenames:
    if fname not in st.session_state["download_selection"]:
        st.session_state["download_selection"][fname] = False

# Remove stale filenames (from old uploads)
for old in list(st.session_state["download_selection"].keys()):
    if old not in input_filenames:
        del st.session_state["download_selection"][old]

# ============================================================
#                ENSEMBLE PREDICTION
# ============================================================

def predict_mask_ensemble(img, models_dict, configs_dict):
    """
    Returns:
      (mask_image_pil, mask_array_uint8[0/1])
    """
    # Resize to a common size for ensemble (use first model's size)
    first_model = list(models_dict.keys())[0]
    common_size = configs_dict[first_model]["input_size"]
    
    # Prepare image for each model
    ensemble = np.zeros(common_size, dtype=np.float32)
    
    with torch.no_grad():
        for mname, model in models_dict.items():
            input_size = configs_dict[mname]["input_size"]
            device = configs_dict[mname]["device"]
            weight = configs_dict[mname]["weight"]
            
            # Resize image to model's input size
            img_resized = img.resize((input_size[1], input_size[0]))
            t = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)
            
            # Run inference
            pred = model(t).cpu().numpy()
            
            # Handle different output shapes
            if pred.ndim == 4:
                pred = pred[0, 0]  # Assuming shape [1, 1, H, W]
            elif pred.ndim == 3:
                pred = pred[0]  # Assuming shape [1, H, W]
            
            # Resize prediction to common size for ensemble
            if pred.shape != common_size:
                pred_resized = np.array(Image.fromarray(pred).resize(
                    (common_size[1], common_size[0]), 
                    resample=Image.BILINEAR
                ))
            else:
                pred_resized = pred
            
            # Apply model weight
            ensemble += pred_resized * weight
    
    # Threshold to create binary mask
    pred_mask = (ensemble > 0.5).astype(np.uint8)
    return Image.fromarray(pred_mask * 255).convert("L"), pred_mask

# Run ensemble for all inputs
pred_results = [predict_mask_ensemble(img, loaded_models, model_configs) for img in input_images]
predicted_images = [r[0] for r in pred_results]
predicted_arrays = [r[1] for r in pred_results]

# ============================================================
#                        METRICS
# ============================================================

def compute_metrics(mask_arr, original_img):
    total_pixels = mask_arr.size
    affected_pixels = np.sum(mask_arr)

    # 1) Affected area %
    affected_pct = (affected_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # 2) Avg intensity inside lesion
    # Resize to mask size for accurate intensity calculation
    gray = original_img.resize((mask_arr.shape[1], mask_arr.shape[0])).convert("L")
    gray_np = np.array(gray)
    avg_intensity = float(gray_np[mask_arr == 1].mean()) if affected_pixels > 0 else 0.0

    # 3) Circularity
    perimeter = np.sum(ndimage.binary_dilation(mask_arr) ^ mask_arr)
    circularity = (4 * math.pi * affected_pixels) / (perimeter ** 2) if perimeter > 0 else 0.0

    # 4) Severity
    # ---- HIGH Severity ----
    if (affected_pct >= 35 and circularity <= 0.45) or (affected_pct >= 45) or (circularity <= 0.30):
        severity="High"

    # ---- MODERATE Severity ----
    elif (15 <= affected_pct < 35) or (0.45 < circularity <= 0.65) or (affected_pct >= 20 and circularity <= 0.6):
        severity="Moderate"

    # ---- LOW Severity ----
    elif affected_pct < 15 and circularity > 0.65:
        severity="Low"
    
    #default case
    else:
        severity="Moderate"

    return affected_pct, avg_intensity, circularity, severity

# ============================================================
#                    HEATMAP OVERLAY
# ============================================================

def create_heatmap_overlay(original_img, mask_arr, intensity=0.4):
    # Get mask size
    mask_h, mask_w = mask_arr.shape
    
    # Resize original to mask size
    orig = np.array(original_img.resize((mask_w, mask_h))).astype(np.uint8)
    if len(orig.shape) == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    # Create colored overlay
    overlay = np.zeros_like(orig)

    # Green = non-affected (mask == 0)
    overlay[mask_arr == 0] = [0, 255, 0]

    # Red = affected (mask == 1)
    overlay[mask_arr == 1] = [255, 0, 0]

    # Blend heatmap with original image
    blended = cv2.addWeighted(orig, 1 - intensity, overlay, intensity, 0)

    return Image.fromarray(blended)

# ============================================================
#                    PDF REPORT BUILDER
# ============================================================

def generate_pdf_report(original_img, mask_img, fname, metrics):
    affected_pct, avg_intensity, circularity, severity = metrics

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>MEDICAL IMAGE ANALYSIS REPORT</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>File:</b> {fname}", styles['Heading3']))
    story.append(Spacer(1, 12))

    # ReportLab needs file paths
    orig_path = "temp_original.png"
    mask_path = "temp_mask.png"
    
    # Resize for PDF display
    display_size = (512, 512)
    original_img.resize(display_size).save(orig_path)
    mask_img.resize(display_size).save(mask_path)

    story.append(Paragraph("<b>Original Image</b>", styles['Heading4']))
    story.append(RLImage(orig_path, width=3*inch, height=3*inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Predicted Mask</b>", styles['Heading4']))
    story.append(RLImage(mask_path, width=3*inch, height=3*inch))
    story.append(Spacer(1, 16))

    # Metrics
    text = f"""
    <b>Affected Area:</b> {affected_pct:.2f}% <br/>
    <b>Avg. Intensity Inside Lesion:</b> {avg_intensity:.2f} <br/>
    <b>Circularity Score:</b> {circularity:.3f} <br/>
    <b>Severity Classification:</b> {severity} <br/>
    """
    story.append(Paragraph("<b>Quantitative Metrics</b>", styles["Heading3"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 12))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer

# For mapping later (CSV/ZIP selected)
by_name = {
    fname: (inp, pred_img, pred_arr)
    for fname, inp, pred_img, pred_arr in zip(
        input_filenames, input_images, predicted_images, predicted_arrays
    )
}

# ============================================================
#                 BATCH EXPORTS (SELECTED ONLY) — TOP
# ============================================================

# Build selection dictionary if needed
if "download_selection" not in st.session_state:
    st.session_state["download_selection"] = {}

# Sync dictionary with current filenames
for fname in input_filenames:
    st.session_state["download_selection"].setdefault(fname, False)

for old in list(st.session_state["download_selection"].keys()):
    if old not in input_filenames:
        del st.session_state["download_selection"][old]

selected_files = [f for f, sel in st.session_state["download_selection"].items() if sel]

st.markdown("### 📦 Batch Exports (Selected Images)")
batch_cols = st.columns(3)

# ---- CSV (selected) ----
with batch_cols[0]:
    if selected_files:
        def generate_csv(selected_files, by_name):
            rows = []
            for fname in selected_files:
                inp, pred_img, pred_arr = by_name[fname]
                m = compute_metrics(pred_arr, inp)
                rows.append({
                    "filename": fname,
                    "affected_pct": round(m[0], 2),
                    "avg_intensity": round(m[1], 2),
                    "circularity": round(m[2], 3),
                    "severity": m[3]
                })
            df = pd.DataFrame(rows)
            return df.to_csv(index=False).encode("utf-8")

        csv_data = generate_csv(selected_files, by_name)
        st.download_button(
            "📊 Download CSV (Selected)",
            data=csv_data,
            file_name="selected_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("📊 Download CSV (Selected)", disabled=True, use_container_width=True)

# ---- Masks ZIP (selected) ----
with batch_cols[1]:
    if selected_files:
        masks_zip = io.BytesIO()
        with zipfile.ZipFile(masks_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in selected_files:
                _, pred_img, _ = by_name[fname]
                buf = io.BytesIO()
                mask_name = fname.rsplit(".", 1)[0] + "_mask.png"
                pred_img.save(buf, format="PNG")
                zipf.writestr(mask_name, buf.getvalue())
        masks_zip.seek(0)
        st.download_button(
            "📦 Download Masks (Selected)",
            data=masks_zip.getvalue(),
            file_name="selected_masks.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.button("📦 Download Masks (Selected)", disabled=True, use_container_width=True)

# ---- Reports ZIP (selected) ----
with batch_cols[2]:
    if selected_files:
        reports_zip = io.BytesIO()
        with zipfile.ZipFile(reports_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in selected_files:
                inp, pred_img, pred_arr = by_name[fname]
                m = compute_metrics(pred_arr, inp)
                pdf_report = generate_pdf_report(inp, pred_img, fname, m)
                report_name = fname.rsplit(".", 1)[0] + "_report.pdf"
                zipf.writestr(report_name, pdf_report.getvalue())
        reports_zip.seek(0)

        st.download_button(
            "📄 Download Reports (Selected)",
            data=reports_zip.getvalue(),
            file_name="selected_reports.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.button("📄 Download Reports (Selected)", disabled=True, use_container_width=True)

st.markdown("---")

# ============================================================
#                 IMAGE CARD CSS (hover + auto height)
# ============================================================

st.markdown("""
<style>
.image-card {
    background: rgba(255, 255, 255, 0.12);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.20);
    transition: all 0.25s ease;
    width: 100%;
    text-align: center;
    height: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.image-card:hover {
    box-shadow: 0 0 20px rgba(102, 252, 241, 0.35);
    transform: translateY(-4px);
}
.image-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
#                        DISPLAY LOOP
# ============================================================

st.subheader("Original • Mask • Heatmap")

# Selection state for batch downloads
if "download_selection" not in st.session_state:
    st.session_state["download_selection"] = {fname: False for fname in input_filenames}

for inp, pred_img, pred_arr, fname in zip(input_images, predicted_images, predicted_arrays, input_filenames):
    # ---- Heatmap intensity slider (per image) ----
    intensity = st.slider(
        f"Heatmap Intensity – {fname}",
        min_value=0.0, max_value=1.0, value=0.40, step=0.05,
        key=f"slider_{fname}"
    )
    heatmap_img = create_heatmap_overlay(inp, pred_arr, intensity=intensity)

    # ---- Three image cards on one row ----
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.markdown("<div class='image-title'>Original Image</div>", unsafe_allow_html=True)
        # Display at mask size
        display_size = (pred_arr.shape[1], pred_arr.shape[0])
        inp_display = inp.resize(display_size)
        st.image(inp_display)

    with col2:
        st.markdown("<div class='image-title'>Predicted Mask</div>", unsafe_allow_html=True)
        st.image(pred_img)
    with col3:
        st.markdown(f"<div class='image-title'>🔥 Heatmap (Intensity {intensity:.2f})</div>", unsafe_allow_html=True)
        st.image(heatmap_img)

    # ---------------------- METRICS DISPLAY --------------------------
    metrics = compute_metrics(pred_arr, inp)
    affected_pct, avg_intensity, circularity, severity = metrics

    st.markdown(
        f"""
        <div style='font-size:17px; line-height:1.6; margin-top:10px;'>
            <b>🩺 Affected Area:</b> {affected_pct:.2f}% <br>
            <b>🌗 Avg. Intensity Inside Lesion:</b> {avg_intensity:.2f} <br>
            <b>⭕ Circularity:</b> {circularity:.3f} <br>
            <b>🧪 Severity Classification:</b> 
            <span style='color:#00eaff; font-weight:700;'>{severity}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------- Downloads + Select ----------------------
    # Mask download
    outname = fname.rsplit(".", 1)[0] + "_mask.png"
    buf = io.BytesIO()
    pred_img.save(buf, format="PNG")
    st.download_button(
        label=f"Download {outname}",
        data=buf.getvalue(),
        file_name=outname,
        mime="image/png",
        key=f"dl_mask_{fname}"
    )

    # PDF report download (per image)
    pdf_buf = generate_pdf_report(inp, pred_img, fname, metrics)
    st.download_button(
        label="📄 Generate PDF Report",
        data=pdf_buf,
        file_name=f"{fname.rsplit('.',1)[0]}_report.pdf",
        mime="application/pdf",
        key=f"dl_pdf_{fname}"
    )

    # Select for batch operations
    st.session_state["download_selection"][fname] = st.checkbox(
        "Select for batch download",
        value=st.session_state["download_selection"][fname],
        key=f"sel_{fname}"
    )

    st.markdown("<hr>", unsafe_allow_html=True)