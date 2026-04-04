import os
import json
import socket
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import gemini_brain

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Robot's UDP Address
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

@app.route('/')
def index():
    """Serves the Web Dashboard Interface"""
    return send_file(os.path.join(os.path.dirname(__file__), 'index.html'))

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Endpoint to receive a video, extract kinematics via Gemini,
    and push the command sequence to the robot over UDP.
    """
    temp_video_path = "temp_web_upload.mp4"
    commands = []

    try:
        if 'file' not in request.files and 'video' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files.get('file') or request.files.get('video')
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file.save(temp_video_path)
        print(f"\n📂 Video saved temporarily as {temp_video_path}")

        print("🧠 Calling Gemini Brain for motion compilation...")
        commands = gemini_brain.compile_motion(temp_video_path)

        if not commands:
            raise ValueError("Gemini failed to generate kinematics. Check API or prompt.")

        # --- 🐛 START DEBUG OUTPUT ---
        print("\n" + "="*40)
        print(" 🐛 DEBUG: AI KINEMATIC COMMANDS ")
        print("="*40)
        print(json.dumps(commands, indent=4))
        print("="*40 + "\n")
        # --- END DEBUG OUTPUT ---

        print("📤 Sending parsed kinematic sequence via UDP...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        response_data = json.dumps(commands).encode('utf-8')
        sock.sendto(response_data, (UDP_IP, UDP_PORT))
        print(f"🎯 Commands successfully dispatched to {UDP_IP}:{UDP_PORT}")

        return jsonify({
            "status": "success",
            "message": "Motion compiled and sent to robot!",
            "commands_generated": len(commands)
        }), 200

    except Exception as e:
        print(f"❌ Error during /upload processing: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                print(f"🗑️ Cleaned up temporary file.\n")
            except Exception as cleanup_err:
                print(f"⚠️ Failed to delete temporary file {temp_video_path}: {cleanup_err}\n")

if __name__ == '__main__':
    print("===================================================")
    print(" 🚀 MOTION MEMORY CORE - SERVER ONLINE ")
    print(" 🌐 Open your browser to: http://127.0.0.1:5000 ")
    print("===================================================")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)