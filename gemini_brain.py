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

SYSTEM_PROMPT = """You are a kinematic compiler for a Pepper robot. 
Watch the human waving BOTH hands. Translate their complex 3D motion into motor arrays.

### JOINT NAMES (Use these exactly):
- Right Arm: RShoulderPitch, RShoulderRoll, RElbowRoll, RHand
- Left Arm: LShoulderPitch, LShoulderRoll, LElbowRoll, LHand
- Head: HeadYaw, HeadPitch

### CRITICAL RULES:
1. Output ONLY a valid JSON array of objects.
2. If BOTH hands move, include BOTH Left and Right joints in the SAME object.
3. Keep the number of joints per object under 6 for stability.
4. Angles must be floats between -2.0 and 2.0.

Example for two-handed wave:
[
  {
    "joints": ["RShoulderPitch", "LShoulderPitch", "RShoulderRoll", "LShoulderRoll"],
    "angles": [-1.0, -1.0, -0.5, 0.5],
    "duration": 1.0
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