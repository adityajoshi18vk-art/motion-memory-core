"""
Violence Filter Module
Extracts the middle frame from a video and runs it through the
violence classification model. Returns a verdict dict.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# ── Model Loading (one-time at import) ────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_violence_classifier_finetuned.keras')
_model = None


def _get_model():
    """Lazy-load the Keras model so the heavy import only happens once."""
    global _model
    if _model is None:
        print("🛡️  Loading violence detection model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("🛡️  Violence detection model loaded successfully.")
    return _model


# ── Frame Extraction ──────────────────────────────────────────────────────────

def extract_middle_frame(video_path: str) -> np.ndarray | None:
    """
    Opens a video file and returns the frame at the temporal midpoint.
    Returns the frame as a BGR numpy array, or None on failure.

    Uses sequential reading because seeking is unreliable for WebM/VP8/VP9
    containers on Windows.  Reads up to MAX_SAMPLE frames then picks the
    middle one, which is fast enough for short motion clips (< 10 s).
    """
    MAX_SAMPLE = 500  # plenty for a 5s @ 30fps clip

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️  Could not open video: {video_path}")
        return None

    # --- Attempt 1: fast seek (works for mp4 / avi) -----------------------
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        mid_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_index)
        ret, frame = cap.read()
        if ret:
            cap.release()
            print(f"🛡️  Extracted frame {mid_index}/{total_frames} (fast seek)")
            return frame
        # seek failed – fall through to sequential read
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Attempt 2: sequential read (robust for webm) ---------------------
    frames = []
    while len(frames) < MAX_SAMPLE:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("⚠️  Video has zero readable frames.")
        return None

    mid = len(frames) // 2
    print(f"🛡️  Extracted frame {mid}/{len(frames)} (sequential read)")
    return frames[mid]


# ── Classification ────────────────────────────────────────────────────────────

def classify_frame(frame: np.ndarray) -> dict:
    """
    Runs the violence classifier on a single BGR frame.

    Returns
    -------
    dict  with keys:
        "is_violent" : bool
        "confidence" : float  (0-100, always refers to the predicted label)
        "label"      : str    ("VIOLENT" or "NON-VIOLENT")
        "raw_score"  : float  (model sigmoid output, >0.5 ⇒ violent)
    """
    model = _get_model()

    # Pre-process to match training pipeline (224×224 RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = img.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x, verbose=0)
    score = float(prediction[0][0])

    is_violent = score > 0.5
    confidence = score * 100 if is_violent else (1 - score) * 100
    label = "VIOLENT" if is_violent else "NON-VIOLENT"

    return {
        "is_violent": is_violent,
        "confidence": round(confidence, 2),
        "label": label,
        "raw_score": round(score, 4),
    }


# ── Public API (the one web_server calls) ────────────────────────────────────

def check_video(video_path: str) -> dict:
    """
    End-to-end check: extract middle frame → classify → return verdict.

    Returns
    -------
    dict  with keys:
        "passed"     : bool   (True if the video is NON-VIOLENT)
        "label"      : str
        "confidence" : float
        "raw_score"  : float
        "error"      : str | None
    """
    print(f"🛡️  Running violence filter on: {video_path}")

    frame = extract_middle_frame(video_path)
    if frame is None:
        return {
            "passed": False,
            "label": "ERROR",
            "confidence": 0.0,
            "raw_score": -1.0,
            "error": "Could not extract a frame from the video for safety screening.",
        }

    result = classify_frame(frame)
    passed = not result["is_violent"]

    verdict_icon = "✅" if passed else "⛔"
    print(f"🛡️  Verdict: {verdict_icon} {result['label']} ({result['confidence']}% confidence)")

    return {
        "passed": passed,
        "label": result["label"],
        "confidence": result["confidence"],
        "raw_score": result["raw_score"],
        "error": None,
    }
