import os
import json
import socket
import time
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import gemini_brain
import violence_filter

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Robot's UDP Address
UDP_IP = "127.0.0.1"
UDP_PORT = 5005


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
    Endpoint to receive a video.  Returns an SSE stream so the frontend
    can show real-time progress through each pipeline stage.
    """
    # ── 1. Validate the incoming file ─────────────────────────────────
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

    # ── 2. Stream the pipeline stages as SSE ──────────────────────────
    def generate():
        commands = []
        try:
            # Stage 1 — File received
            yield _sse("progress", {
                "step": 1,
                "icon": "📂",
                "message": "Video uploaded successfully"
            })

            # Stage 2 — Safety screening
            yield _sse("progress", {
                "step": 2,
                "icon": "🛡️",
                "message": "Running safety screening..."
            })

            verdict = violence_filter.check_video(temp_video_path)

            if not verdict["passed"]:
                error_detail = verdict.get("error")
                if error_detail:
                    msg = error_detail
                else:
                    msg = (
                        f"Video rejected: classified as {verdict['label']} "
                        f"({verdict['confidence']}% confidence). "
                        "Please upload a non-violent motion video."
                    )
                print(f"⛔ Upload blocked by safety filter: {msg}")
                yield _sse("rejected", {
                    "message": msg,
                    "filter": {
                        "label": verdict["label"],
                        "confidence": verdict["confidence"],
                        "raw_score": verdict["raw_score"],
                    },
                })
                return

            yield _sse("progress", {
                "step": 3,
                "icon": "✅",
                "message": f"Safety check passed — NON-VIOLENT ({verdict['confidence']}%)"
            })

            # Stage 3 — Gemini compilation
            yield _sse("progress", {
                "step": 4,
                "icon": "🧠",
                "message": "Compiling motion via Gemini AI..."
            })

            commands = gemini_brain.compile_motion(temp_video_path)

            if not commands:
                yield _sse("error", {
                    "message": "Gemini failed to generate kinematics. Check API key or prompt."
                })
                return

            yield _sse("progress", {
                "step": 5,
                "icon": "⚙️",
                "message": f"Generated {len(commands)} kinematic keyframes"
            })

            # Debug output
            print("\n" + "="*40)
            print(" 🐛 DEBUG: AI KINEMATIC COMMANDS ")
            print("="*40)
            print(json.dumps(commands, indent=4))
            print("="*40 + "\n")

            # Stage 4 — UDP dispatch
            yield _sse("progress", {
                "step": 6,
                "icon": "📤",
                "message": "Sending commands to robot via UDP..."
            })

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            response_data = json.dumps(commands).encode('utf-8')
            sock.sendto(response_data, (UDP_IP, UDP_PORT))
            print(f"🎯 Commands dispatched to {UDP_IP}:{UDP_PORT}")

            # Stage 5 — Done
            yield _sse("done", {
                "message": "Motion compiled and sent to robot!",
                "commands_generated": len(commands),
            })

        except Exception as e:
            print(f"❌ Error during /upload processing: {str(e)}")
            yield _sse("error", {"message": str(e)})

        finally:
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    print(f"🗑️ Cleaned up temporary file.\n")
                except Exception as cleanup_err:
                    print(f"⚠️ Failed to delete temp file: {cleanup_err}\n")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


if __name__ == '__main__':
    # Pre-load the model at startup so the first request isn't slow
    print("===================================================")
    print(" 🚀 MOTION MEMORY CORE - SERVER ONLINE ")
    print(" 🌐 Open your browser to: http://127.0.0.1:5000 ")
    print("===================================================")
    print(" 🛡️  Pre-loading violence detection model...")
    violence_filter._get_model()
    print("===================================================")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)