import os
import subprocess
import threading
import sys
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Load API key
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for the dashboard

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/trigger_learning', methods=['POST'])
def trigger_learning():
    """
    Orchestrates the Motion Memory Pipeline:
    1. Receive Uploaded Video
    2. Analyze Video (mediapipe_engine.py)
    3. Return JSON sequence
    """
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        video_path = "demo_motion.webm"
        file.save(video_path)
        print(f"  🎬  Saved incoming video chunk to {video_path}")

        # Use the python executable from the .venv if possible
        python_exe = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe')
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        # 2. Analyze Video (mediapipe_engine.py)
        print("  🤖  Starting motion analysis (mediapipe_engine.py)...")
        analysis_proc = subprocess.run(
            [python_exe, "mediapipe_engine.py", video_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        if analysis_proc.returncode != 0:
            print(f"  ❌  Analysis FAILED: {analysis_proc.stderr}")
            return jsonify({
                "error": "Analysis failed",
                "stderr": analysis_proc.stderr
            }), 500

        # Parse the output to find the JSON result
        # Our mediapipe_engine.py prints the sequence to stdout
        import json
        output = analysis_proc.stdout
        
        # Extract everything from the last '[' to the last ']'
        try:
            start_idx = output.rfind('[')
            end_idx = output.rfind(']') + 1
            json_str = output[start_idx:end_idx]
            commands = json.loads(json_str)
        except Exception as e:
            return jsonify({
                "error": "Failed to parse model output",
                "raw_output": output,
                "exception": str(e)
            }), 500

        print(f"  ✅  Success! Compiled {len(commands)} commands.")
        return jsonify({
            "status": "success",
            "model": "mediapipe-pose-engine",
            "commands": commands,
            "video_path": video_path
        })

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "exception": str(e)
        }), 500

if __name__ == '__main__':
    print("  🚀  Motion Memory Backend started on http://localhost:5000")
    print("  👉  Connect your dashboard to this endpoint.")
    # Threaded mode to allow UI interaction while subprocesses run
    app.run(port=5000, threaded=True)
