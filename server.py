import os
import subprocess
import threading
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load API key
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for the dashboard

@app.route('/trigger_learning', methods=['GET'])
def trigger_learning():
    """
    Orchestrates the Motion Memory Pipeline:
    1. Record Video (vision.py)
    2. Analyze Video (gemini_brain.py)
    3. Return JSON sequence
    """
    try:
        # Use the python executable from the .venv if possible
        python_exe = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe')
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        # 1. Record Video (vision.py)
        print("  🎬  Starting video recording (vision.py)...")
        record_proc = subprocess.run(
            [python_exe, "vision.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
        )
        if record_proc.returncode != 0:
            print(f"  ❌  Recording FAILED: {record_proc.stderr}")
            return jsonify({
                "error": "Recording failed",
                "stderr": record_proc.stderr
            }), 500

        # 2. Analyze Video (gemini_brain.py)
        print("  🤖  Starting motion analysis (gemini_brain.py)...")
        analysis_proc = subprocess.run(
            [python_exe, "gemini_brain.py", "demo_motion.mp4"],
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
        # Our gemini_brain.py prints the sequence at the end
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
            "model": "gemini-2.5-flash",
            "commands": commands,
            "video_path": "demo_motion.mp4"
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
