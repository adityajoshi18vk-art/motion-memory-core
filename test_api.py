import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def test_key():
    print("--- TESTING API KEY COMPATIBILITY ---")
    if not GEMINI_API_KEY:
        print("ERROR: No GEMINI_API_KEY found in .env file!")
        return

    print("Key length: " + str(len(GEMINI_API_KEY)))
    
    try:
        # Initialize client using the new google.genai SDK patterns
        # Note: If this fails, the SDK or Key is incompatible
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        print("Connecting to Google AI services...")
        # Simplest way to test: List available models or generate tiny text
        response = client.models.generate_content(
            model='models/gemini-2.5-flash', 
            contents="say 'API KEY IS COMPATIBLE'"
        )
        
        if response and response.text:
            output = response.text.strip()
            print("SUCCESS! AI says: " + str(output))
            print("Your key is valid and compatible.")
        else:
            print("STALEMATE: Response received but text was empty.")

    except Exception as e:
        print("COMPATIBILITY FAILED!")
        print("Error Details: " + str(e).encode('ascii', 'ignore').decode())

if __name__ == "__main__":
    test_key()
