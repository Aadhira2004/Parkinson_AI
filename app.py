import streamlit as st
import av
import cv2
import math
import numpy as np
import time
import os

# MacOS M4 Stability Patch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# ---------------- PAGE CONFIG & THEME ----------------
st.set_page_config(page_title="AI Parkinson's Screening", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Text Visibility Fix */
    html, body, [class*="css"], .stMarkdown, p, li, label {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF !important;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Clinical Metric Styling */
    [data-testid="stMetricValue"] {
        color: #007BFF !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] p {
        color: #A0AEC0 !important;
    }

    /* Custom Header Card */
    .header-card {
        background: linear-gradient(90deg, #1A202C 0%, #0F1116 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #007BFF;
        margin-bottom: 25px;
    }

    .college-name {
        color: #A0AEC0 !important;
        font-size: 14px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .project-title {
        color: #FFFFFF !important;
        font-size: 32px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Checkbox & Button Visibility */
    .stCheckbox label { color: #FFFFFF !important; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #007BFF;
        color: white !important;
        font-weight: 600;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- EYE INDEX CONSTANTS ----------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices):
    v1 = math.hypot(landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x,
                    landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y)
    v2 = math.hypot(landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x,
                    landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y)
    h = math.hypot(landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x,
                   landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y)
    return (v1 + v2) / (2.0 * h)

# ---------------- VIDEO PROCESSOR ----------------
class BlinkProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.blinks = 0
        self.eye_closed = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark
            avg_ear = (calculate_ear(mesh, LEFT_EYE) + calculate_ear(mesh, RIGHT_EYE)) / 2.0
            if avg_ear < 0.21: self.eye_closed = True
            elif avg_ear > 0.26 and self.eye_closed:
                self.blinks += 1
                self.eye_closed = False
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- INSTITUTIONAL HEADER ----------------
st.markdown(f"""
    <div class="header-card">
        <div class="college-name">Rohini College of Engineering and Technology</div>
        <div class="project-title">AI Parkinson's Screening System</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR: RESEARCH TEAM ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("Research Team")
    
    with st.expander("👥 Team Members", expanded=True):
        st.write("**Aadhira Suleim A R** \n(963322121001)")
        st.write("**Jeswin Joe S** \n(963322121031)")
        st.write("**Elakkiya K** \n(963322121018)")
        st.write("**Indhuja D** \n(963322121026)")
    
    with st.expander("🎓 Project Guide", expanded=True):
        st.write("**Dr. R.B. Benisha**")
        st.caption("Biomedical Engineering")
    
    st.divider()
    st.caption("Department of Biomedical Engineering")
    st.caption("RCET - 2026")

# ---------------- APP LOGIC FLOW ----------------
if "step" not in st.session_state:
    st.session_state.step = "scan"

if st.session_state.step == "scan":
    st.subheader("Stage 1: 60-Second Ocular Motor Analysis")
    timer_placeholder = st.empty()
    
    ctx = webrtc_streamer(
        key="blink-detection",
        video_processor_factory=BlinkProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        if "start_time" not in st.session_state: st.session_state.start_time = time.time()
        elapsed = int(time.time() - st.session_state.start_time)
        remaining = max(0, 60 - elapsed)
        timer_placeholder.metric("Time Remaining", f"{remaining}s")
        st.progress(min(elapsed / 60, 1.0))
        if elapsed >= 60:
            st.session_state.blinks = ctx.video_processor.blinks
            st.session_state.step = "questions"
            st.rerun()
        else:
            time.sleep(0.1)
            st.rerun()

elif st.session_state.step == "questions":
    st.subheader("Stage 2: Non-Motor Symptom Assessment")
    col1, col2 = st.columns(2)
    with col1:
        s1 = st.checkbox("Vivid/Violent dreams")
        s2 = st.checkbox("Talking in sleep")
        s3 = st.checkbox("Daytime fatigue")
    with col2:
        o1 = st.checkbox("Loss of smell")
        o2 = st.checkbox("Bland food taste")
        o3 = st.checkbox("Coffee/Smoke detection issues")

    if st.button("Generate Final Report"):
        st.session_state.total_symptoms = sum([s1, s2, s3, o1, o2, o3])
        st.session_state.step = "report"
        st.rerun()

elif st.session_state.step == "report":
    st.header("📋 Clinical Screening Report")
    c1, c2 = st.columns(2)
    c1.metric("Blink Rate", f"{st.session_state.blinks} BPM")
    c2.metric("Symptom Score", f"{st.session_state.total_symptoms}/6")

    if st.session_state.blinks < 12 and st.session_state.total_symptoms >= 3:
        st.error("High Clinical Correlation Detected")
    else:
        st.success("Normal Baseline Detected")

    if st.button("Reset Test"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
