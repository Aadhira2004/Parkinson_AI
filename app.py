import streamlit as st
import av
import cv2
import math
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch

# Direct imports for macOS/M4 stability
import mediapipe as mp
from mediapipe.solutions import face_mesh as mp_face_mesh

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Neurovision AI", page_icon="🧠", layout="wide")

# Eye landmark indices
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
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
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

            if avg_ear < 0.20:
                self.eye_closed = True
            elif avg_ear > 0.25 and self.eye_closed:
                self.blinks += 1
                self.eye_closed = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- APP LOGIC ----------------
if "step" not in st.session_state:
    st.session_state.step = "scan"

st.title("🧠 Neurovision AI – Clinical Screening")

if st.session_state.step == "scan":
    st.subheader("Stage 1: 60-Second Ocular Motor Analysis")
    st.info("Ensure your face is well-lit. The timer starts when the camera activates.")
    
    timer_placeholder = st.empty()
    
    ctx = webrtc_streamer(
        key="blink-task",
        video_processor_factory=BlinkProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if ctx.video_processor:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()
        
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
        st.markdown("**Sleep Patterns**")
        s1 = st.checkbox("Vivid or disturbing dreams")
        s2 = st.checkbox("Talking or acting out dreams")
        s3 = st.checkbox("Restless legs at night")
    with col2:
        st.markdown("**Olfactory Health**")
        o1 = st.checkbox("Noticeable loss of smell")
        o2 = st.checkbox("Food tastes bland/different")
        o3 = st.checkbox("Trouble smelling coffee/smoke")

    if st.button("Generate Clinical Analysis"):
        st.session_state.total_score = sum([s1, s2, s3, o1, o2, o3])
        st.session_state.step = "report"
        st.rerun()

elif st.session_state.step == "report":
    bpm = st.session_state.blinks
    score = st.session_state.total_score

    st.header("📋 Clinical Analysis Report")
    
    c1, c2 = st.columns(2)
    c1.metric("Blink Rate", f"{bpm} BPM")
    c2.metric("Symptom Score", f"{score}/6")

    if bpm < 12 and score >= 3:
        st.error("Correlation: High. Please consult a neurological specialist.")
    elif bpm < 15 or score >= 2:
        st.warning("Correlation: Moderate. Consider follow-up monitoring.")
    else:
        st.success("Correlation: Low. Results are within typical range.")

    if st.button("Start New Session"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
