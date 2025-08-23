import os
import uuid
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# Force CPU-only for DeepFace/TensorFlow/PyTorch backends
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from services.face_matcher import (
    load_target_embedding,
    match_faces_in_video,
    ModelAndBackend,
)


UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "static", "outputs")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def ensure_directories() -> None:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def create_app() -> Flask:
    ensure_directories()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
    
    # Increase file upload limits
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
    app.config['MAX_CONTENT_PATH'] = None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/test")
    def test_upload_page():
        return render_template("test_upload.html")

    @app.post("/test")
    def test_upload():
        print("=== TEST UPLOAD START ===")
        print(f"Request method: {request.method}")
        print(f"Request content type: {request.content_type}")
        print(f"Request content length: {request.content_length}")
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        
        face_file = request.files.get("face_image")
        video_file = request.files.get("video_file")
        
        print(f"Face file: {face_file}")
        print(f"Video file: {video_file}")
        
        if face_file:
            print(f"Face file filename: {face_file.filename}")
            print(f"Face file size: {len(face_file.read())}")
            face_file.seek(0)  # Reset file pointer
            
        if video_file:
            print(f"Video file filename: {video_file.filename}")
            print(f"Video file size: {len(video_file.read())}")
            video_file.seek(0)  # Reset file pointer
            
        print("=== TEST UPLOAD END ===")
        return f"Files received: Face={face_file.filename if face_file else 'None'}, Video={video_file.filename if video_file else 'None'}"

    @app.post("/process")
    def process():
        print("=== PROCESS REQUEST START ===")
        print(f"Request method: {request.method}")
        print(f"Request content type: {request.content_type}")
        print(f"Request content length: {request.content_length}")
        print(f"Request files: {request.files}")
        print(f"Request form: {request.form}")
        print(f"Request headers: {dict(request.headers)}")
        
        # Check if request is too large
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            print(f"ERROR: Request too large: {request.content_length} bytes")
            flash("File too large. Please use smaller files.")
            return redirect(url_for("index"))
        
        # Validate files
        face_file = request.files.get("face_image")
        video_file = request.files.get("video_file")
        
        print(f"Face file: {face_file}")
        print(f"Video file: {video_file}")
        
        if face_file:
            print(f"Face file filename: {face_file.filename}")
            print(f"Face file content type: {face_file.content_type}")
        if video_file:
            print(f"Video file filename: {video_file.filename}")
            print(f"Video file content type: {video_file.content_type}")
        
        if not face_file or face_file.filename == "":
            print("ERROR: No face image uploaded")
            flash("Please upload a face image.")
            return redirect(url_for("index"))
        if not video_file or video_file.filename == "":
            print("ERROR: No video file uploaded")
            flash("Please upload a video file.")
            return redirect(url_for("index"))

        print(f"Face filename: {face_file.filename}")
        print(f"Video filename: {video_file.filename}")

        if not allowed_file(face_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            print(f"ERROR: Invalid face image format: {face_file.filename}")
            flash("Unsupported face image format. Use png, jpg, or jpeg.")
            return redirect(url_for("index"))
        if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
            print(f"ERROR: Invalid video format: {video_file.filename}")
            flash("Unsupported video format. Use mp4, avi, mov, or mkv.")
            return redirect(url_for("index"))

        # Read options
        try:
            threshold = float(request.form.get("threshold", "0.75"))
        except ValueError:
            threshold = 0.75
        try:
            frame_skip = int(request.form.get("frame_skip", "1"))
            frame_skip = max(1, frame_skip)
        except ValueError:
            frame_skip = 1
        try:
            resize_width = int(request.form.get("resize_width", "640"))
            resize_width = max(0, resize_width)
        except ValueError:
            resize_width = 640

        model_name = request.form.get("model", "ArcFace")
        detector_backend = request.form.get("backend", "mtcnn")
        
        print(f"Options: threshold={threshold}, frame_skip={frame_skip}, resize_width={resize_width}")
        print(f"Model: {model_name}, Backend: {detector_backend}")

        # Generate run ID
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        print(f"Run ID: {run_id}")
        
        # Persist uploads
        run_upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], run_id)
        run_output_dir = os.path.join(app.config["OUTPUT_FOLDER"], run_id)
        os.makedirs(run_upload_dir, exist_ok=True)
        os.makedirs(run_output_dir, exist_ok=True)

        face_filename = secure_filename(face_file.filename)
        video_filename = secure_filename(video_file.filename)
        face_path = os.path.join(run_upload_dir, face_filename)
        video_path = os.path.join(run_upload_dir, video_filename)
        
        print(f"Saving face to: {face_path}")
        print(f"Saving video to: {video_path}")
        
        face_file.save(face_path)
        video_file.save(video_path)
        
        print("Files saved successfully")

        # Compute target embedding
        print("Starting face embedding extraction...")
        model_backend = ModelAndBackend(model_name=model_name, detector_backend=detector_backend)
        embedding, backend_used = load_target_embedding(face_path, model_backend)
        
        if embedding is None:
            print("ERROR: Failed to extract face embedding")
            flash("Failed to extract a face embedding from the uploaded image.")
            return redirect(url_for("index"))
        
        print(f"Face embedding extracted successfully using backend: {backend_used}")

        # Match faces in video
        print("Starting video processing...")
        matches, meta = match_faces_in_video(
            video_path=video_path,
            target_embedding=embedding,
            model_backend=model_backend,
            preferred_backend=backend_used,
            similarity_threshold=threshold,
            frame_skip=frame_skip,
            resize_width=resize_width,
            output_dir=run_output_dir,
        )
        
        print(f"Video processing complete. Found {len(matches)} matches")

        # Build URLs for results page
        rel_output_dir = os.path.join("outputs", run_id)
        for m in matches:
            # Convert absolute path to url path under static
            if m.get("saved_path"):
                filename = os.path.basename(m["saved_path"])
                m["url"] = url_for("static", filename=os.path.join(rel_output_dir, filename))

        print("=== PROCESS REQUEST COMPLETE ===")
        return render_template(
            "results.html",
            matches=matches,
            meta=meta,
            threshold=threshold,
            frame_skip=frame_skip,
            resize_width=resize_width,
            model=model_name,
            backend=backend_used,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


