import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables (API Key)
load_dotenv()

# ── YOUR API KEY ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Master Roboticist and Biomechanics AI programming a SoftBank Pepper Robot. 
Your objective is to translate human video motion into flawless, physically possible kinematic keyframes.

### ⚙️ THE HARDWARE (15 Degrees of Freedom & Limits in Radians)
HEAD:
- HeadYaw (Turn L/R): 1.5 (Left) to -1.5 (Right)
- HeadPitch (Up/Down): -0.6 (Up) to 0.5 (Down)

SHOULDERS:
- LShoulderPitch/RShoulderPitch: -2.0 (Raised straight up) to 2.0 (Straight down by side)
- LShoulderRoll: 0.1 (By side) to 1.5 (Max lift left/T-Pose)
- RShoulderRoll: -0.1 (By side) to -1.5 (Max lift right/T-Pose)

ELBOWS (Crucial for complex poses):
- LElbowYaw/RElbowYaw: -2.0 (Twisted inward across chest) to 2.0 (Twisted outward away from body)
- LElbowRoll: -1.5 (Fully bent tightly) to 0.0 (Straight)
- RElbowRoll: 0.0 (Straight) to 1.5 (Fully bent tightly)

WRISTS & HANDS:
- LWristYaw/RWristYaw: -1.8 (Palm down) to 1.8 (Palm up)
- LHand/RHand: 0.0 (Closed) to 1.0 (Open)

TORSO:
- HipPitch: -0.8 (Bowing/Leaning) to 0.0 (Standing straight)

### 🧠 ADVANCED HEURISTICS (Maximum Flexibility & Fluidity)
1. FULL RANGE OF MOTION: You are completely unbound and free to compose ANY human gesture you observe. Use the absolute limits of the hardware when the human stretches fully (e.g., reaching maximum upward extent at -2.0 ShoulderPitch), deeply bends, points, or bows.
2. COMPLEX FREEFORM POSES: The robot must faithfully mimic saluting, folded arms, dancing, leaning, outward reaching, etc. Dynamically calculate the perfect combination of Pitch, Roll, and Yaw across all joints to fluidly achieve the target 3D position.
3. CREATIVE INTERPOLATION: Do not follow rigid step-by-step rules. If the human performs a fast, dynamic, or highly unusual gesture, compose the most accurate combination of joint angles to mirror it in reality.
4. KINEMATIC SAFETY & MICRO-STEPPING: Break sweeping or complex motions into as many chronological keyframes as needed (e.g., 3-5 frames) for smooth, safe motor interpolation. Avoid clipping arms through the Torso or Head by utilizing ShoulderRoll and ElbowYaw.

### 📝 REQUIRED OUTPUT FORMAT
You must first write a brief `<THOUGHT_PROCESS>` analyzing the physics of the human's movement, identifying self-collision risks, and planning the micro-steps. 

Then, output ONLY a valid, minified JSON array of objects wrapped in markdown ```json ``` blocks. 

Example Output:
<THOUGHT_PROCESS>
The user is saluting. To avoid collision, I must first raise the RShoulderPitch and RElbowRoll, then in the next frame, apply negative RElbowYaw to twist the hand inward to the forehead. I will set the duration to 0.6s per frame for fluidity.
</THOUGHT_PROCESS>
```json
[
  {"joints": ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"], "angles": [-1.0, -0.3, 1.2], "duration": 0.6},
  {"joints": ["RElbowYaw", "RWristYaw", "RHand"], "angles": [-1.5, -1.0, 1.0], "duration": 0.6}
]
"""

def compile_motion(video_path):
    print("  🚀  Connecting to Gemini 2.5 Flash...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        print(f"  🎬  Uploading '{video_path}'...")
        video_file = client.files.upload(file=video_path)
        
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Google's servers failed to process this video file format!")

        print("\n  🤖  Generating Dynamic Kinematics (God Mode)...")
        
        # --- THE FIX: PROPER SYSTEM INSTRUCTION & TEMPERATURE ---
        unfiltered_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,  
            temperature=0.2,                   # Forces precise JSON math
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[video_file], 
            config=unfiltered_config
        )
        
        if getattr(response, 'text', None) is None:
            raise ValueError(f"AI returned NO text! Raw Debug Info: {response}")
            
        raw_text = response.text.strip()
        clean_text = raw_text
        
        if "```" in raw_text:
            parts = raw_text.split("```")
            for part in parts:
                if "json" in part.lower() or "[" in part:
                    clean_text = part.replace("json", "").strip()
                    break
        else:
            clean_text = raw_text

        commands = json.loads(clean_text)

        for cmd in commands:
            if "duration" not in cmd:
                cmd["duration"] = 1.0

        print(f"  ✅  Successfully compiled {len(commands)} dynamic movements.")
        return commands
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌  AI Pipeline Error: {error_msg}")
        
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            print("  💡  API Quota Exceeded! Please wait 60 seconds, and ensure your video is under 3 seconds long.")
            return []
            
        return []