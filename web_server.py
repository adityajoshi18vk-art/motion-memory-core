import os
import sys
import contextlib

# 🤫 SILENCE THE TRASH (Must be at the very top)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import socket
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import gemini_brain
import violence_filter

# Mute the internal TensorFlow logger
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
CORS(app)

# Robot's UDP Address
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# --- 🛡️ STRICT MODEL BOOT CHECK ---
print("\n" + "="*50)
print(" 🛡️  BOOTING MOTION MEMORY CORE (STRICT AI MODE) ")
print(" ⏳ Muffling terminal and verifying Neural Classifier...")

# Mandatory startup check as requested by USER
try:
    # Use contextlib to catch any sneak residue from Keras 
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
             # This call triggers the model load in violence_filter.py
             violence_filter._get_model()
    
    print(" ✅ AI MODEL LOADED & VERIFIED AT STARTUP")
    print(" ✅ PRE-FILTER SYSTEM: ACTIVE")
    print("="*50 + "\n")
except Exception as e:
    print("\n" + "!"*50)
    print(f"❌ CRITICAL ERROR: Model failed to initialize!")
    print(f"🔍 Detail: {e}")
    print("!"*50 + "\n")
    sys.exit(1)

def _sse(event, data):
    """Format a single Server-Sent Event line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.route('/')
def index():
    """Serves the Web Dashboard Interface"""
    return send_file(os.path.join(os.path.dirname(__file__), 'index.html'))

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Endpoint to receive a video. Returns an SSE stream so the frontend
    can show real-time progress through each pipeline stage.
    """
    # 1. Validate the incoming file
    if 'file' not in request.files and 'video' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files.get('file') or request.files.get('video')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Preserve original extension so OpenCV picks the right decoder
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    temp_video_path = f"temp_web_upload{ext}"
    file.save(temp_video_path)
    print(f"\n📂 Video saved temporarily as {temp_video_path}")

    # 2. Stream the pipeline stages as SSE
    def generate():
        try:
            # Stage 1 — File received
            yield _sse("progress", {
                "step": 1,
                "icon": "📂",
                "message": "Video uploaded successfully"
            })

            # Stage 2 — AI Safety screening
            yield _sse("progress", {
                "step": 2,
                "icon": "🛡️",
                "message": "Running safety screening..."
            })

            verdict = violence_filter.check_video(temp_video_path)

            if not verdict["passed"]:
                msg = verdict.get("error") or (
                    f"Video rejected: classified as {verdict.get('label', 'VIOLENT')} "
                    f"({verdict.get('confidence', 0)}% confidence). "
                    "Please record a safe motion."
                )
                print(f"⛔ Rejection: {msg}")
                yield _sse("rejected", {"message": msg})
                return

            yield _sse("progress", {
                "step": 3,
                "icon": "✅",
                "message": f"Safety check passed ({verdict['confidence']}% certainty)"
            })

            # Stage 3 — Gemini compilation
            yield _sse("progress", {
                "step": 4,
                "icon": "🧠",
                "message": "Compiling motion via Gemini AI..."
            })

            commands = gemini_brain.compile_motion(temp_video_path)

            if not commands:
                yield _sse("error", {"message": "Gemini generation failed."})
                return

            yield _sse("progress", {
                "step": 5,
                "icon": "⚙️",
                "message": f"Compiled {len(commands)} keyframes"
            })

            # Stage 4 — UDP dispatch
            yield _sse("progress", {
                "step": 6,
                "icon": "📤",
                "message": "Dispatching to robot..."
            })

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            response_data = json.dumps(commands).encode('utf-8')
            sock.sendto(response_data, (UDP_IP, UDP_PORT))
            print(f"🎯 Commands dispatched to {UDP_IP}:{UDP_PORT}")

            # Stage 5 — Done
            yield _sse("done", {
                "message": "Motion verified and sent!",
                "commands_generated": len(commands),
            })

        except Exception as e:
            print(f"❌ Error during /upload processing: {str(e)}")
            yield _sse("error", {"message": str(e)})

        finally:
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )

if __name__ == '__main__':
    print(" 🚀 SERVER GOING LIVE AT http://127.0.0.1:5000 ")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
