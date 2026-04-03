# Pepper-Automator 🤖

A robotic kinematic compiler using Gemini 2.5 Flash to translate human motion into motor arrays for the Pepper robot.

## Features
- **Real-time Recording**: Capture human motion via OpenCV.
- **AI Compilation**: Multimodal video analysis to extract joint angles.
- **Pepper Robot Support**: Targeted joint mapping for `RShoulderPitch`, `LShoulderPitch`, etc.
- **Secure Integration**: API keys managed via environment variables.

## Setup
1. Create a `.env` file and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_key_here
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   python server.py
   ```

## Development
Initialized as `pepper-automator`. Built with Python, OpenCV, and Google GenAI.
