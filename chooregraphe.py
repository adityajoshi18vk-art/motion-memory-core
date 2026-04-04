class MyClass(GeneratedClass):
    def __init__(self):
        GeneratedClass.__init__(self)

    def onLoad(self):
        import socket
        import json
        import time
        self.socket = socket
        self.json = json
        self.time = time

    def onInput_onStart(self):
        UDP_IP = "127.0.0.1"
        UDP_PORT = 5005

        try:
            self.logger.info("Requesting dynamic motion from AI...")
            sock = self.socket.socket(self.socket.AF_INET, self.socket.SOCK_DGRAM)
            sock.settimeout(60.0) # Wait up to 60s for Gemini 2.5

            sock.sendto("WAKE_UP", (UDP_IP, UDP_PORT))

            data, addr = sock.recvfrom(4096)
            commands = self.json.loads(data)

            motion = ALProxy("ALMotion")

            # --- THE AUTOMATION LOOP ---
            for cmd in commands:
                joints = [str(j) for j in cmd["joints"]] # Ensure strings for Python 2.7
                angles = [float(a) for a in cmd["angles"]]
                duration = float(cmd.get("duration", 1.0))

                self.logger.info("Moving: " + str(joints))

                # Execute the math Gemini calculated
                motion.setAngles(joints, angles, 0.2)
                self.time.sleep(duration)

            self.onStopped()

        except Exception as e:
            self.logger.error("Automation Failed: " + str(e))
            self.onStopped()