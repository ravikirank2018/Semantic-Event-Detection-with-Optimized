import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from detect_events import EventDetector
import time

# Page Config
st.set_page_config(page_title="Semantic Event Detection", page_icon="📹", layout="wide")

st.title("📹 Semantic Event Detection with Optimized VLM")
st.markdown("""
Upload a video to detect semantic events:
- **Person Walking**
- **Vehicle Stopping**
- **Crowded Scene**
""")

# Sidebar
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
model_type = st.sidebar.selectbox("Model", ["YOLO-World (Baseline)"])

# File Uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Video")
        st.video(video_path)

    if st.button("Run Detection"):
        st.text("Processing... Please wait.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize Detector
        detector = EventDetector() # Uses default parameters
        
        # Output Setup
        output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Streamlit requires H264 for browser playback usually. 
        # OpenCV's default mp4v might not play in all browsers locally without re-encoding.
        # We'll try 'avc1' or just save and hope the browser supports the container. 
        # For full web compatibility we might need ffmpeg, but valid mp4 usually works.
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        events_log = []
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            processed_frame, events = detector.detect_events(frame)
            
            # Log events
            if events:
                timestamp = frame_idx / fps
                for event in events:
                    events_log.append({
                        "Time (s)": f"{timestamp:.2f}",
                        "Frame": frame_idx,
                        "Event": event
                    })
            
            out.write(processed_frame)
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress = frame_idx / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing Frame {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        progress_bar.progress(1.0)
        status_text.text("Processing Complete!")
        
        # Display Results
        with col2:
            st.subheader("Processed Video")
            # Re-open the file to ensure it's ready
            if os.path.exists(output_path):
                # We often need to re-encode for streamlit to show it properly if codec issues arise.
                # Since we accept we might not have ffmpeg, we attempt to show it directly.
                st.video(output_path)
            else:
                st.error("Output file creation failed.")
        
        # Event Table
        st.subheader("Detected Events Log")
        if events_log:
            df = pd.DataFrame(events_log)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Event Log (CSV)",
                csv,
                "events.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No specific events (Walking/Stopping/Crowd) detected with current thresholds.")

