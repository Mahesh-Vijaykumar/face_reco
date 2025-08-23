# Face Recognition in Video - Project Explanation

## 📋 Project Overview

**Purpose**: Real-time face recognition from a video against a user-uploaded target face image.

**Key Features**:
- CPU-only execution (no GPU required)
- Upload target image and video
- Adjustable threshold, frame skip, and resize
- Accurate embeddings (default ArcFace) with robust detectors (MTCNN/RetinaFace)
- Saves and displays all matched frames with bounding boxes and similarity
- Modern Flask-based web interface (no Streamlit)

## 🏗️ Architecture

### Frontend (Flask Templates)
- **`templates/index.html`**: Upload form and controls
- **`templates/results.html`**: Grid gallery of matched frames with similarity and time
- **`static/styles.css`**: Aesthetic dark theme, responsive layout

### Backend (Flask App + Service)
- **`app.py`**: Routes, file handling, CPU-only env, calling the face matcher
- **`services/face_matcher.py`**: Embedding extraction, per-frame detection + matching, saving annotated frames

## 🛠️ Tech Stack and Tools

### Core Technologies
- **Web Framework**: Flask 3.0.3
- **Computer Vision**: OpenCV 4.11.0
- **Face Recognition**: DeepFace 0.0.93
  - Models: ArcFace, Facenet, VGG-Face
  - Detectors: MTCNN, RetinaFace, SSD, OpenCV
- **Numerics**: NumPy 2.1.3, SciPy 1.15.3
- **Image Processing**: Pillow 11.2.1
- **Runtime**: Python 3.10+ (works on Linux/Windows)
- **Optional CPU acceleration**: onnxruntime 1.17.3

### Dependencies
- **mtcnn**: 1.0.0 (CPU-friendly face detector)
- **retina-face**: 0.0.17 (accurate face detector)
- **gunicorn**: 23.0.0 (production WSGI server)

## 🔄 How It Works (Code Flow)

### 1. Route `GET /` (in `app.py`)
- Renders `index.html` with:
  - File inputs: `face_image`, `video_file`
  - Options: `threshold`, `frame_skip`, `resize_width`, `model`, `backend`

### 2. Route `POST /process` (in `app.py`)
- Validates inputs and saves to `uploads/<run_id>/`
- Reads options (defaults: threshold 0.75, frame skip 1, resize width 640, model ArcFace, backend MTCNN)
- Calls `load_target_embedding(face_path, model_backend)` to compute the target embedding with fallback backends
- Calls `match_faces_in_video(...)` to:
  - Open video with OpenCV
  - Optionally resize frames and skip frames for performance
  - Detect faces and represent embeddings per frame via DeepFace
  - Compare cosine similarity against target embedding
  - For matches, draw bounding boxes + label and save annotated frame in `static/outputs/<run_id>/`
- Returns `results.html` with all matches and metadata (FPS, processed frames, etc.)

### 3. `services/face_matcher.py`
- **`ModelAndBackend`**: Dataclass holding `model_name` and `detector_backend`
- **`load_target_embedding(path, model_backend)`**: Returns embedding and detector actually used (tries fallbacks)
- **`match_faces_in_video(...)`**:
  - Iterates frames, uses DeepFace with selected/backup detector backends
  - Computes similarity: `1 - cosine(target, frame_face_embedding)`
  - Saves annotated image for each match via `_annotate_and_save(...)`
  - Returns list of matches: frame number, timestamp seconds, similarity, saved image path, backend

### 4. CPU-only Safeguards
- `CUDA_VISIBLE_DEVICES=-1` and `TF_CPP_MIN_LOG_LEVEL=2` set in `app.py`
- Default detector is `MTCNN` to avoid heavy GPU-specific backends by default (you can switch in UI)

## ⚡ Accuracy and Performance Tips

### Accuracy Optimization
- Use ArcFace + RetinaFace for strongest performance
- Raise threshold (e.g., 0.80–0.90) if needed
- Provide a clear, frontal target face image

### Performance (CPU)
- Increase frame skip (e.g., 2–5) for faster processing
- Lower resize width (e.g., 480–640) for speed
- Use MTCNN or OpenCV detector for CPU-friendly processing
- Switch to RetinaFace if accuracy needed

## 📁 Outputs

### Where Results Are Saved
- **Path**: `static/outputs/<run_id>/match_fXXXXXX_<time>s.jpg`
- **Content**: Annotated frames with bounding box, similarity score, and timestamp

### What You See in UI
- Grid gallery of all matched frames
- Each frame shows: similarity score, timestamp, frame number
- Bounding boxes drawn around detected faces

## 🚀 Installation and Setup

### Prerequisites
- Python 3.10 or higher
- Git (for cloning)
- Internet connection (for downloading models)

### Cloning from GitHub

#### Linux/macOS:
```bash
git clone YOUR_REPO_URL.git
cd facereco
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

#### Windows (PowerShell):
```powershell
git clone YOUR_REPO_URL.git
cd facereco
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

#### If Port 5000 is Busy:
```bash
# Linux/macOS
PORT=5001 python app.py

# Windows PowerShell
$env:PORT=5001; python app.py
```

## 📂 Project Structure
```
facereco/
├── app.py                    # Flask app, routes, upload handling, options, CPU-only env
├── services/
│   ├── __init__.py          # Makes services a package
│   └── face_matcher.py      # Core matching logic, embedding extraction, frame processing
├── templates/
│   ├── index.html           # Upload form and controls
│   └── results.html         # Match gallery with similarities and timestamps
├── static/
│   └── styles.css           # UI styles
├── uploads/                 # Created at runtime, stores uploaded face/video
├── static/outputs/          # Created at runtime, stores annotated match frames
├── requirements.txt         # Python dependencies
├── README.md               # Basic setup instructions
└── EXPLANATION.md          # This comprehensive guide
```

## 🔧 Environment Variables (Optional)
- **`PORT`**: Set a custom port (default 5000)
- **`SECRET_KEY`**: Flask secret key (default dev key)

## 🐛 Common Issues and Solutions

### TensorFlow/CUDA Warnings
- **Issue**: CUDA-related warnings in terminal
- **Solution**: Safe to ignore; app runs CPU-only as intended
- **Note**: These are just TensorFlow trying to use GPU libraries that aren't available

### Model Downloads Slow
- **Issue**: First run takes time to download face recognition models
- **Solution**: Switch detector to `mtcnn` or `opencv` in the UI for faster startup

### No Matches Found
- **Issue**: App doesn't find any face matches
- **Solutions**:
  - Lower similarity threshold (try 0.60-0.70)
  - Improve target image quality (clear, frontal face)
  - Try different detector backend
  - Ensure target face is actually in the video

### Port Already in Use
- **Issue**: "Address already in use" error
- **Solution**: Use different port: `PORT=5001 python app.py`

## 🎯 Quick Start (Most Common)

### Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Windows:
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## 📊 Performance Benchmarks

### Typical Processing Times (CPU-only)
- **Small video (30s, 640px width)**: 2-5 minutes
- **Medium video (2min, 640px width)**: 8-15 minutes
- **Large video (5min, 640px width)**: 20-40 minutes

### Memory Usage
- **Base memory**: ~500MB
- **During processing**: 1-2GB (depends on video size)

### Optimization Tips
- Use frame skip of 2-3 for faster processing
- Reduce resize width to 480px for speed
- Use MTCNN detector for CPU efficiency

## 🔒 Security and Privacy

### Data Handling
- All processing is local; no data leaves your machine
- Uploaded files are stored temporarily in `uploads/` directory
- Processed results are saved in `static/outputs/` directory
- No external API calls or cloud processing

### File Cleanup
- Consider implementing automatic cleanup of old uploads/outputs
- Manual cleanup: delete contents of `uploads/` and `static/outputs/` directories

## 🚀 Production Deployment

### Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:create_app()
```

### Environment Variables for Production
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secure-secret-key
export PORT=5000
```

### Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📚 Additional Resources

### DeepFace Documentation
- [DeepFace GitHub](https://github.com/serengil/deepface)
- [Available Models](https://github.com/serengil/deepface/wiki/Models)
- [Detector Backends](https://github.com/serengil/deepface/wiki/Detector-Backends)

### Flask Documentation
- [Flask Official Docs](https://flask.palletsprojects.com/)
- [Flask Deployment](https://flask.palletsprojects.com/en/2.3.x/deploying/)

### OpenCV Documentation
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Testing
- Test with different video formats (MP4, AVI, MOV)
- Test with various image formats (JPG, PNG)
- Test with different face detection scenarios
- Verify CPU-only operation

