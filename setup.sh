#!/bin/bash

# --- 1. COLOR PRESETS ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Neurovision AI Setup...${NC}"

# --- 2. CREATE VIRTUAL ENVIRONMENT ---
echo "Creating Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

# --- 3. INSTALL DEPENDENCIES ---
echo "Installing requirements (OpenCV, MediaPipe, Streamlit)..."
pip install --upgrade pip
pip install streamlit opencv-python mediapipe numpy

# --- 4. LAUNCH APP ---
echo -e "${GREEN}Setup Complete! Launching Neurovision AI...${NC}"
streamlit run app.py