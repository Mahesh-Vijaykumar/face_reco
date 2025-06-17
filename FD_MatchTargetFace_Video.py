import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from PIL import Image
import tempfile
import warnings
warnings.filterwarnings("ignore")

st.title("🎯 Face Recognition in Video (Upload Target Face Image)")

def get_face_embedding(image, model='Facenet', backends=['mtcnn', 'retinaface', 'opencv', 'ssd']):
    for backend in backends:
        try:
            rep = DeepFace.represent(image, model_name=model, detector_backend=backend)
            return rep[0]['embedding'], backend
        except:
            continue
    return None, None

def match_face_in_video(video_path, target_embedding, backend_used, threshold=0.6):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_number = 0
    matching_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        try:
            result = DeepFace.represent(frame, model_name='Facenet', detector_backend=backend_used)
            for entry in result:
                emb = entry['embedding']
                similarity = 1 - cosine(target_embedding, emb)
                if similarity > threshold:
                    stframe.image(frame, channels='BGR', caption=f"✅ Match in frame {frame_number} (similarity = {similarity:.2f})")
                    matching_frames.append(frame_number)
                    break
        except:
            continue

    cap.release()
    return matching_frames

face_file = st.file_uploader("🖼️ Upload Face Image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("📹 Upload Video File", type=["mp4", "avi", "mov"])

if face_file and video_file:
    face_image = np.array(Image.open(face_file))
    st.image(face_image, caption="Target Face", use_container_width=True)

    st.markdown("🔍 Extracting face embedding...")
    embedding, backend_used = get_face_embedding(face_image)

    if embedding is None:
        st.error("❌ Failed to extract face embedding from image.")
    else:
        st.success(f"✅ Face embedding extracted using `{backend_used}`")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        st.markdown("🔄 Matching face in video...")
        matches = match_face_in_video(video_path, embedding, backend_used)

        if matches:
            st.success(f"🎉 Face matched in {len(matches)} frame(s): {matches}")
        else:
            st.error("❌ Face not found in the video.")
