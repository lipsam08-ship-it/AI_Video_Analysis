import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Title of the Dashboard
st.title("🎥 AI Video Analytics Dashboard")
st.write("Upload a video to analyze objects using AI ")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # small model for speed, use yolov8m.pt or yolov8l.pt for accuracy

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)  
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Show video info
    st.subheader("📹 Video Information")
    st.text(f"Resolution: {width}x{height}")
    st.text(f"Frame Rate: {fps} fps")
    st.text(f"Total Frames: {total_frames}")

    # Read first frame
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        st.subheader("🖼️ First Frame Preview")
        st.image(pil_img, width=600)

        # Run AI inference
        if st.button("🔍 Detect with AI"):
            with st.spinner("Analyzing..."):
                results = model.predict(img, conf=0.5, classes=[0])  # class 0 = person

                # Plot result
                annotated = results[0].plot()  # numpy array with boxes
                st.subheader("🎯 AI Detection Results")
                st.image(annotated, caption="Detected Persons", width=600)

                # Show detected labels
                labels = results[0].boxes.cls.cpu().numpy()
                names = [model.names[int(i)] for i in labels]
                st.write("**Detected Objects:**", ", ".join(set(names)))

    cap.release()
else:
    st.info("Please upload a video file to begin.")
