import socket
import json
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = b"WAKE_UP"

print(f"DEBUG: KNOCKING ON {UDP_IP}:{UDP_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(120.0)

try:
    start_time = time.time()
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    print("Trigger sent. Waiting for AI kinematics...")
    
    data, addr = sock.recvfrom(65535) # Using Large buffer
    elapsed = time.time() - start_time
    print(f"Received {len(data)} bytes in {elapsed:.2f}s from {addr}")
    
    commands = json.loads(data)
    print(f"Decoded {len(commands)} commands.")
    print("Sample:", commands[0] if commands else "None")

except socket.timeout:
    print("ERROR: Timeout! Gemini did not respond within 120s.")
except Exception as e:
    print(f"ERROR: {e}")
finally:
    sock.close()
