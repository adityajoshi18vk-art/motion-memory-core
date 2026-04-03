import socket
import json
import gemini_brain

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

print(f"🚀 UDP Server starting on {UDP_IP}:{UDP_PORT}...")

# Create the UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    print("Listening for Choregraphe trigger...")
    
    # 1. Wait for the robot to send a message to port 5005
    data, addr = sock.recvfrom(1024) 
    print(f"Received trigger from robot at {addr}!")

    try:
        # 2. Trigger the AI compiler
        print("Compiling video via Gemini...")
        commands = gemini_brain.compile_motion("demo_motion.mp4")

        # 3. Convert to JSON bytes and send it back to the exact same address
        response_data = json.dumps(commands).encode('utf-8')
        sock.sendto(response_data, addr)
        print("✅ Commands sent back via UDP successfully!\n")
        
    except Exception as e:
        print(f"❌ Error compiling: {e}")
        sock.sendto(b"[]", addr) # Send an empty array so the robot doesn't crash