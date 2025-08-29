
Face Recognition in Video (Flask, CPU-only)

A modern Flask web app that matches a target face in an uploaded video. Runs fully on CPU with DeepFace for embeddings and OpenCV for video processing. Outputs all matched frames with bounding boxes and similarity scores.

Features
- Upload target face and video
- CPU-only execution (ArcFace + RetinaFace by default)
- Adjustable similarity threshold, frame skip, and resize width
- Shows all matched frames with annotated bounding boxes and similarity
- Clean, responsive UI

Requirements
- Python 3.10+
- Linux/macOS/Windows

Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run
```bash
python app.py
# Open http://localhost:5000
```

Options (UI)
- Similarity threshold: 0.50–0.95 (default 0.75)
- Process every Nth frame: 1–10 (default 1)
- Resize width: 0 (disabled) or pixels (default 640)
- Model: ArcFace, Facenet, VGG-Face
- Detector backend: retinaface, mtcnn, ssd, opencv

Notes
- All processing is local; no data leaves your machine.
- For large videos, increase frame skip or reduce resize width for speed.

Project structure
```
facereco/
├─ app.py
├─ services/
│  ├─ __init__.py
│  └─ face_matcher.py
├─ templates/
│  ├─ index.html
│  └─ results.html
├─ static/
│  └─ styles.css
├─ uploads/           # created at runtime
└─ static/outputs/    # created at runtime
```

Troubleshooting
- If RetinaFace model downloads are slow, switch to MTCNN or OpenCV in the UI.
- If accuracy is low, try ArcFace + RetinaFace with a higher threshold (0.80+).


This project is a real-time face recognition app that uses **DeepFace** for face embeddings and **OpenCV** for video processing. It allows you to:

- Upload a **target face image**
- Upload a **video file**
- Automatically **detect and match** the face in video frames
- Preview matched frames and show similarity scores
- Works entirely on **CPU** (with optimizations)

---

##  Features

-  Upload face image and video
-  Uses DeepFace (Facenet or Dlib)
- Optimized for low-memory systems
  - Skips frames
  - Resizes frames
  - Optional: Limits max frame count
-  Displays matched frames with confidence
-  Lists all matched frame numbers

---


##  How It Works

1. Extracts 128D embedding from the uploaded face image
2. Iterates through frames of the uploaded video
3. Runs DeepFace embedding on detected faces
4. Computes cosine similarity and highlights matches

---

##  Tips

- Works best with clear, frontal face images
- Lower the threshold in code (`similarity > 0.6`) if too strict
- For large videos, limit frame count or increase skip rate

---


##  License
created by Mahesh-Vijaykumar


