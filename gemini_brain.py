import os
import json
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables (API Key)
load_dotenv()

# ── YOUR API KEY ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a master kinematic compiler for a SoftBank Pepper robot.
Carefully analyze the video of the human performing movements. Translate their complete sequence of 3D motions into precise robotic motor command arrays.

### JOINT NAMES (Use these exactly for mapping):
- Head: HeadYaw, HeadPitch
- Right Arm: RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw, RHand
- Left Arm: LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw, LHand
- Torso/Lower: HipRoll, HipPitch, KneePitch

### CRITICAL RULES:
1. Output ONLY a valid JSON array of objects representing chronological keyframes.
2. Capture ALL joint movements seen in the video natively. Do not isolate just the arms if the head or torso moves.
3. Combine all simultaneous joint movements into the SAME keyframe object to ensure fluid multi-limb motion.
4. Scale human angles appropriately. Angles must be floats generally between -2.0 and 2.0 (radians). Hands use 0.0 (closed) to 1.0 (open).
5. Provide a realistic execution `duration` (in seconds) for each keyframe to simulate authentic human speed.

Example for a complex body routine:
[
  {
    "joints": ["HeadPitch", "RShoulderPitch", "RHand", "HipPitch"],
    "angles": [-0.2, -1.5, 1.0, -0.3],
    "duration": 1.2
  },
  {
    "joints": ["HeadYaw", "LShoulderPitch", "LShoulderRoll", "LElbowRoll", "RHand", "HipPitch"],
    "angles": [0.5, -1.2, 0.4, -0.8, 0.0, 0.0],
    "duration": 0.8
  }
]
"""

def compile_motion(video_path):
    print("\n  🚀  Connecting to Gemini 2.5...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    try:
        print(f"  🎬  Uploading '{video_path}'...")
        video_file = client.files.upload(file=video_path)
        
        # Wait for the vision pipeline to finish processing the video
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        print("\n  🤖  Generating Dynamic Kinematics (Dual-Arm Support)...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[video_file, SYSTEM_PROMPT]
        )
        
        # --- THE FIX: ROBUST JSON EXTRACTION ---
        raw_text = response.text.strip()
        
        # Remove any Markdown code blocks if Gemini included them
        if "```" in raw_text:
            # Splits by ``` and takes the part inside the blocks
            parts = raw_text.split("```")
            for part in parts:
                if "json" in part.lower() or "[" in part:
                    clean_text = part.replace("json", "").strip()
                    break
        else:
            clean_text = raw_text

        commands = json.loads(clean_text)

        # Safety check: Ensure every command has a duration for the robot loop
        for cmd in commands:
            if "duration" not in cmd:
                cmd["duration"] = 1.0

        print(f"  ✅  Successfully compiled {len(commands)} dynamic movements.")
        return commands
        
    except json.JSONDecodeError as je:
        print(f"  ❌  JSON formatting error from AI: {je}")
        print(f"  Raw Output was: {raw_text[:100]}...") # Print start of error for debugging
        return []
    except Exception as e:
        print(f"  ❌  AI Pipeline Error: {e}")
        return []