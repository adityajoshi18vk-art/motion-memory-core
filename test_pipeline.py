import json, time, os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load key from .env file
load_dotenv()
KEY = os.getenv("GEMINI_API_KEY")
ALLOWED = ["RAISE_RIGHT_ARM", "LOWER_RIGHT_ARM", "OPEN_HAND", "CLOSE_HAND", "TURN_TORSO_LEFT"]
PROMPT  = (
    "You are a robotic kinematic compiler. Watch the video and output ONLY a JSON array "
    f"using these exact command strings: {json.dumps(ALLOWED)}. "
    "Empty array [] if nothing maps. No markdown, no explanation."
)

client = genai.Client(api_key=KEY)

# Upload
print("Uploading demo_motion.mp4 via File API...")
with open("demo_motion.mp4", "rb") as f:
    uf = client.files.upload(file=f, config=types.UploadFileConfig(mime_type="video/mp4"))
print(f"Uploaded: {uf.name}  state={uf.state.name}")

# Poll via files.list() until ACTIVE
print("Waiting for ACTIVE state", end="", flush=True)
target_name = uf.name
video_file = None
for _ in range(30):
    time.sleep(3)
    for fi in client.files.list():
        if fi.name == target_name:
            if fi.state.name == "ACTIVE":
                video_file = fi
            elif fi.state.name == "FAILED":
                print("\nFile processing FAILED.")
                exit(1)
            print(f"[{fi.state.name}]", end="", flush=True)
            break
    if video_file:
        break
print(" Done!" if video_file else "\nTimeout!")

if not video_file:
    exit(1)

# Infer
print("Querying gemini-2.0-flash...")
resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[types.Content(parts=[
        types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type="video/mp4")),
        types.Part(text=PROMPT),
    ])],
)
print("\n✅ Model output:", resp.text.strip())
