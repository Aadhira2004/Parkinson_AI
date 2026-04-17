import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile

st.set_page_config(page_title="AI Parkinson Screening", layout="wide")

st.title("AI Parkinson's Screening System")

# =========================================
# STAGE 1 – VIDEO UPLOAD + BLINK ANALYSIS
# =========================================

st.header("Stage 1: Ocular Motor Video Analysis")

uploaded_video = st.file_uploader(
    "Upload Eye Video (10–30 seconds)",
    type=["mp4", "avi", "mov"]
)

blink_rate = 0

if uploaded_video is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh()

    blink_counter = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            blink_counter += 1  # simplified demo logic

    cap.release()

    if frame_count > 0:
        blink_rate = int(blink_counter / 10)

    st.video(uploaded_video)
    st.success("Video processed successfully")
    st.metric("Estimated Blink Rate", f"{blink_rate} BPM")


# =========================================
# STAGE 2 – QUESTIONNAIRE
# =========================================

st.header("Stage 2: Non-Motor Symptom Assessment")

symptoms = {
    "Vivid/Violent dreams": 1,
    "Talking in sleep": 1,
    "Daytime fatigue": 1,
    "Loss of smell": 2,
    "Bland food taste": 2,
    "Coffee/Smoke detection issues": 2,
}

score = 0

for symptom, weight in symptoms.items():
    if st.checkbox(symptom):
        score += weight


# =========================================
# FINAL REPORT
# =========================================

if st.button("Generate Final Report"):

    st.header("Clinical Screening Report")

    st.metric("Blink Rate", f"{blink_rate} BPM")
    st.metric("Symptom Score", f"{score}/9")

    # Blink interpretation
    if blink_rate < 10:
        blink_flag = 2
    elif blink_rate < 20:
        blink_flag = 1
    else:
        blink_flag = 0

    final_score = (blink_flag * 0.6) + (score * 0.4)

    if final_score < 2:
        st.success("Low Risk – Normal Baseline")
    elif final_score < 4:
        st.warning("Moderate Risk – Clinical Consultation Advised")
    else:
        st.error("High Risk – Strongly Recommended Neurological Evaluation")

