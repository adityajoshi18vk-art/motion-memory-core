import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def list_models():
    print("--- FETCHING AVAILABLE MODELS ---")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for model in client.models.list():
            print(f"Model ID: {model.name}")
        print("\n✅ Key is valid enough to list models.")
    except Exception as e:
        print(f"❌ FAILED to list models: {str(e)}")

if __name__ == "__main__":
    list_models()
