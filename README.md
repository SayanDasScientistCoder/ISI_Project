# 🧠 Automated Medical Image Segmentation & Analytics Platform

This project is a **full-stack, Streamlit-based medical image segmentation system** designed for **research and clinical prototyping**.  
It supports **multi-model weighted ensembling**, **per-model device selection (CPU/GPU)**, **medical metric computation**, **visual analytics**, **PDF report generation**, and **secure user authentication with MongoDB**.

The system is built as a **production-ready prototype**, suitable for academic evaluation, demonstrations, and further research extensions.

---

## 📌 Key Highlights

- Secure **login, registration, and password reset**
- **MongoDB-backed user dashboard** with real usage statistics
- **Multi-model inference** with weighted ensembling
- **Per-model device selection** (CPU / GPU if CUDA available)
- Medical metrics: **area %, circularity, intensity**
- **Severity classification**
- Interactive **heatmap visualization**
- **PDF, CSV, ZIP** report generation
- Clean, modern **Streamlit UI**
- Modular, extensible architecture

---

## 🏗️ System Architecture (High Level)

User
└── Streamlit UI
├── Authentication Layer
│ └── MongoDB Atlas
│
├── Upload Module
│
├── Inference Engine
│ ├── Model Loader
│ ├── Per-model Device Selector (CPU / GPU)
│ ├── Weighted Ensemble
│
├── Analytics Engine
│ ├── Area %
│ ├── Circularity
│ ├── Severity Classification
│
├── Visualization Layer
│ └── Heatmaps & Masks
│
└── Reporting Module
├── PDF Reports
├── CSV Summaries
└── ZIP Exports

## 🗂️ Project Structure

project_root/
│
├── app.py # Entry point and routing
├── config.py # Environment & app configuration
│
├── pages/
│ ├── 0_Login.py # Login & registration
│ ├── 1_Upload.py # Image upload page
│ ├── 2_Result.py # Inference, ensembling & analytics
│ ├── 3_Dashboard.py # User dashboard (MongoDB-backed)
│ └── ResetPassword.py # Password reset workflow
│
├── models/ # Pretrained segmentation models (.pt / .pth)
├── styles/
│ └── theme.css # UI styling
│
├── requirements.txt
├── README.md
└── .env # Environment variables (not committed)


---

## 🔐 Authentication & Security

- User registration and login via **MongoDB Atlas**
- Passwords stored using **bcrypt hashing**
- Password reset via **email-based OTP/token**
- Reset tokens are **time-limited**
- Auth guards protect all internal pages
- Session state cleared on logout

---

## 🧠 Inference & Ensembling Pipeline

1. User uploads one or more images
2. User selects one or more segmentation models
3. User assigns **weights per model**
4. User selects **execution device per model**
   - CPU
   - GPU (CUDA, if available)
5. Each model runs inference independently
6. Model outputs are normalized
7. Weighted ensemble mask is generated
8. Binary thresholding applied
9. Metrics are computed
10. Visualizations and reports are generated

---

## ⚙️ Device Selection Logic

- Each model can independently run on:
  - **CPU**
  - **GPU (CUDA)** if available
- System automatically falls back to CPU if GPU is unavailable
- Mixed-device execution is supported

---

## 📊 Medical Metrics Computed

- **Affected Area Percentage**
- **Average Lesion Intensity**
- **Circularity Score**
- **Severity Classification**

### Severity Levels
| Severity | Criteria |
|--------|---------|
| Low | Small area AND high circularity |
| Moderate | Medium area OR medium circularity |
| High | Large area OR low circularity |

---

## 🎨 Visualization Features

- Original image
- Binary segmentation mask
- Red–green heatmap overlay
- Adjustable opacity
- Interactive preview

---

## 📄 Reporting & Export

- Per-image **PDF medical report**
- Batch exports:
  - CSV metrics summary
  - ZIP of masks
  - ZIP of PDF reports
- User-selected batch downloads

---

## 📊 User Dashboard

The dashboard displays **real, persistent data** from MongoDB:

- User email
- Account creation date
- Last login time
- Total uploads
- Total predictions
- Subscription type (Free / extensible)

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
GMAIL_ADDRESS=your_email@gmail.com
GMAIL_APP_PASSWORD=your_gmail_app_password
