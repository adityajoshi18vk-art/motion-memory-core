import subprocess
import os
import sys

def run_test():
    print("--- 1. RECORDING TEST ---")
    res = subprocess.run([sys.executable, "vision.py"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    print("CODE:", res.returncode)

    if res.returncode != 0:
        return

    print("\n--- 2. ANALYSIS TEST ---")
    res = subprocess.run([sys.executable, "gemini_brain.py", "demo_motion.mp4"], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    print("CODE:", res.returncode)

if __name__ == "__main__":
    run_test()
