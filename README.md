# 🎯 Face Recognition in Video with DeepFace and Streamlit

This project is a real-time face recognition app that uses **DeepFace** for face embeddings and **OpenCV** for video processing. It allows you to:

- Upload a **target face image**
- Upload a **video file**
- Automatically **detect and match** the face in video frames
- Preview matched frames and show similarity scores
- Works entirely on **CPU** (with optimizations)

---

## 📦 Features

- ✅ Upload face image and video
- 🧠 Uses DeepFace (Facenet or Dlib)
- ⚡ Optimized for low-memory systems
  - Skips frames
  - Resizes frames
  - Optional: Limits max frame count
- 📸 Displays matched frames with confidence
- 📊 Lists all matched frame numbers

---

## 📁 Installation

```bash
git clone https://github.com/your-username/face-recognition-streamlit.git
cd face-recognition-streamlit
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run FD_MatchTargetFace_Video_Optimized.py
```

---

## 🧠 Model Backends Used

| Task              | Backend         |
|-------------------|-----------------|
| Face Detection    | MTCNN / RetinaFace / OpenCV / SSD |
| Embedding         | Facenet (default), Dlib (optional) |
| Matching          | Cosine similarity |

---

## 📝 How It Works

1. Extracts 128D embedding from the uploaded face image
2. Iterates through frames of the uploaded video
3. Runs DeepFace embedding on detected faces
4. Computes cosine similarity and highlights matches

---

## 💡 Tips

- Works best with clear, frontal face images
- Lower the threshold in code (`similarity > 0.6`) if too strict
- For large videos, limit frame count or increase skip rate

---

## 📌 TODO

- [ ] Save matched frames
- [ ] Export match logs to CSV
- [ ] Auto-download model files if missing

---

## 📃 License

MIT License

> Created with ❤️ using Streamlit + DeepFace
