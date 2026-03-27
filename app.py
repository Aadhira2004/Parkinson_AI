import streamlit as st
import av
import cv2
import math
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

st.set_page_config(page_title="Neurovision AI", page_icon="🧠", layout="wide")

# ---------------- EYE INDEX ----------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices):
    v1 = math.hypot(
        landmarks[eye_indices[1]].x - landmarks[eye_indices[5]].x,
        landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y
    )
    v2 = math.hypot(
        landmarks[eye_indices[2]].x - landmarks[eye_indices[4]].x,
        landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y
    )
    h = math.hypot(
        landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x,
        landmarks[eye_indices[0]].y - landmarks[eye_indices[3]].y
    )
    return (v1 + v2) / (2.0 * h)

# ---------------- VIDEO PROCESSOR ----------------
class BlinkProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blinks = 0
        self.eye_closed = False
        self.start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark
            avg_ear = (
                calculate_ear(mesh, LEFT_EYE) +
                calculate_ear(mesh, RIGHT_EYE)
            ) / 2.0

            if avg_ear < 0.21:
                self.eye_closed = True
            elif avg_ear > 0.26 and self.eye_closed:
                self.blinks += 1
                self.eye_closed = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- APP ----------------
st.title("🧠 Neurovision AI – Clinical Screening")

if "step" not in st.session_state:
    st.session_state.step = "scan"

# ---------------- STAGE 1 ----------------
if st.session_state.step == "scan":

    st.subheader("Stage 1: 60-Second Ocular Motor Analysis")

    ctx = webrtc_streamer(
        key="blink",
        video_processor_factory=BlinkProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        elapsed = int(time.time() - ctx.video_processor.start_time)
        remaining = max(0, 60 - elapsed)
        st.progress(min(elapsed / 60, 1.0))
        st.write(f"Time Remaining: {remaining} seconds")

        if elapsed >= 60:
            st.session_state.blinks = ctx.video_processor.blinks
            st.session_state.step = "questions"
            st.rerun()

# ---------------- STAGE 2 ----------------
elif st.session_state.step == "questions":

    st.subheader("Stage 2: Non-Motor Symptom Assessment")

    sleep_q = [
        "Vivid dreams",
        "Talking in sleep",
        "Acting out dreams",
        "Daytime sleepiness",
        "Restless legs",
    ]

    smell_q = [
        "Reduced smell",
        "Food tastes bland",
        "Cannot smell coffee",
        "Trouble detecting smoke",
        "General scent loss",
    ]

    s_score = sum([st.checkbox(q) for q in sleep_q])
    o_score = sum([st.checkbox(q) for q in smell_q])

    if st.button("Generate Clinical Report"):
        st.session_state.s_score = s_score
        st.session_state.o_score = o_score
        st.session_state.step = "report"
        st.rerun()

# ---------------- REPORT ----------------
elif st.session_state.step == "report":

    blinks = st.session_state.blinks
    bpm = blinks  # since measured for 60 seconds
    total_symptom = st.session_state.s_score + st.session_state.o_score

    st.header("📋 Clinical Screening Report")

    st.write(f"Blink Rate: {bpm} BPM")
    st.write(f"Symptom Score: {total_symptom}")

    if bpm < 12 and total_symptom >= 4:
        st.error("High Clinical Correlation Detected")
    elif bpm < 15 or total_symptom >= 3:
        st.warning("Moderate Correlation")
    else:
        st.success("Low Correlation")

    st.info("Normal Blink Range: 15–20 BPM")

    # PDF DOWNLOAD
    if st.button("Download PDF Report"):
        doc = SimpleDocTemplate("Neurovision_Report.pdf")
        elements = []

        style = ParagraphStyle(
            name="Normal",
            fontSize=12,
            textColor=colors.black
        )

        elements.append(Paragraph("Neurovision AI Clinical Screening Report", style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Blink Rate: {bpm} BPM", style))
        elements.append(Paragraph(f"Symptom Score: {total_symptom}", style))

        doc.build(elements)

        with open("Neurovision_Report.pdf", "rb") as f:
            st.download_button(
                "Click to Download",
                f,
                file_name="Neurovision_Report.pdf"
            )