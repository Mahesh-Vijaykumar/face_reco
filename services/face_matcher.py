import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine


@dataclass
class ModelAndBackend:
    model_name: str = "ArcFace"
    detector_backend: str = "retinaface"


FALLBACK_BACKENDS: List[str] = ["retinaface", "mtcnn", "ssd", "opencv"]


def _resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / float(width)
    new_w = target_width
    new_h = int(height * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_target_embedding(
    image_path: str, model_backend: ModelAndBackend
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    print(f"Loading target embedding from: {image_path}")
    print(f"Model: {model_backend.model_name}, Backend: {model_backend.detector_backend}")
    
    backends_to_try = [model_backend.detector_backend] + [b for b in FALLBACK_BACKENDS if b != model_backend.detector_backend]
    print(f"Backends to try: {backends_to_try}")
    
    for backend in backends_to_try:
        try:
            print(f"Trying backend: {backend}")
            reps = DeepFace.represent(img_path=image_path, model_name=model_backend.model_name, detector_backend=backend)
            if not reps:
                print(f"No faces detected with backend: {backend}")
                continue
            embedding = np.array(reps[0]["embedding"], dtype=np.float32)
            print(f"Successfully extracted embedding with backend: {backend}")
            return embedding, backend
        except Exception as e:
            print(f"Error with backend {backend}: {str(e)}")
            continue
    
    print("Failed to extract embedding with all backends")
    return None, None


def _annotate_and_save(
    frame_bgr: np.ndarray,
    facial_area: Dict[str, int],
    similarity: float,
    output_dir: str,
    frame_number: int,
    timestamp_seconds: float,
) -> str:
    x = int(facial_area.get("x", 0))
    y = int(facial_area.get("y", 0))
    w = int(facial_area.get("w", 0))
    h = int(facial_area.get("h", 0))
    pt1 = (x, y)
    pt2 = (x + w, y + h)

    annotated = frame_bgr.copy()
    cv2.rectangle(annotated, pt1, pt2, (0, 200, 0), 2)
    label = f"sim {similarity:.2f} | {timestamp_seconds:.2f}s"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated, (x, max(0, y - th - 8)), (x + tw + 6, y), (0, 200, 0), -1)
    cv2.putText(annotated, label, (x + 3, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"match_f{frame_number:06d}_{timestamp_seconds:.2f}s.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, annotated)
    return save_path


def match_faces_in_video(
    video_path: str,
    target_embedding: np.ndarray,
    model_backend: ModelAndBackend,
    preferred_backend: Optional[str] = None,
    similarity_threshold: float = 0.75,
    frame_skip: int = 1,
    resize_width: int = 640,
    output_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"Starting video processing: {video_path}")
    print(f"Similarity threshold: {similarity_threshold}, Frame skip: {frame_skip}, Resize width: {resize_width}")
    
    matches: List[Dict[str, Any]] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return matches, {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    print(f"Video info: FPS={fps}, Total frames={total_frames}")

    backends_to_try = []
    if preferred_backend:
        backends_to_try.append(preferred_backend)
    for b in FALLBACK_BACKENDS:
        if b not in backends_to_try:
            backends_to_try.append(b)

    frame_number = 0
    processed_frames = 0
    matches_found = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1

        if frame_skip > 1 and (frame_number - 1) % frame_skip != 0:
            continue

        processed_frames += 1
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames, found {matches_found} matches so far")
            
        if resize_width > 0:
            frame = _resize_keep_aspect(frame, resize_width)

        represent_success = False
        reps: Optional[List[Dict[str, Any]]] = None
        used_backend: Optional[str] = None

        for backend in backends_to_try:
            try:
                reps = DeepFace.represent(img_path=frame, model_name=model_backend.model_name, detector_backend=backend)
                used_backend = backend
                represent_success = True
                break
            except Exception as e:
                continue

        if not represent_success or not reps:
            continue

        for entry in reps:
            embedding = np.array(entry.get("embedding", []), dtype=np.float32)
            if embedding.size == 0:
                continue
            sim = 1.0 - float(cosine(target_embedding, embedding))
            if sim >= similarity_threshold:
                fa = entry.get("facial_area", {}) or {}
                ts = frame_number / float(fps)
                saved_path = None
                if output_dir:
                    saved_path = _annotate_and_save(frame, fa, sim, output_dir, frame_number, ts)
                matches.append(
                    {
                        "frame": frame_number,
                        "time_seconds": ts,
                        "similarity": sim,
                        "facial_area": fa,
                        "backend": used_backend,
                        "saved_path": saved_path,
                    }
                )
                matches_found += 1

    cap.release()
    
    print(f"Video processing complete. Processed {processed_frames} frames, found {matches_found} matches")

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "duration_seconds": (total_frames / fps) if fps > 0 else None,
        "model": model_backend.model_name,
        "preferred_backend": preferred_backend,
    }
    return matches, meta


