"""
MediaPipe Pose Engine — Full-Body Kinematic Compiler for SoftBank Pepper
Extracts joint-to-joint angles from video using MediaPipe Pose landmarks
and maps them to Pepper's motor command format.
"""
import sys
import json
import math
import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe Pose Landmark Indices ──────────────────────────────────────────
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM_NOSE        = 0
LM_L_EAR       = 7
LM_R_EAR       = 8
LM_L_SHOULDER  = 11
LM_R_SHOULDER  = 12
LM_L_ELBOW     = 13
LM_R_ELBOW     = 14
LM_L_WRIST     = 15
LM_R_WRIST     = 16
LM_L_PINKY     = 17
LM_R_PINKY     = 18
LM_L_INDEX     = 19
LM_R_INDEX     = 20
LM_L_HIP       = 23
LM_R_HIP       = 24

# ── Pepper Joint Limits (radians) ───────────────────────────────────────────
PEPPER_LIMITS = {
    "HeadYaw":        (-2.0857, 2.0857),
    "HeadPitch":      (-0.7068, 0.6371),
    "RShoulderPitch": (-2.0857, 2.0857),
    "RShoulderRoll":  (-1.5620, -0.0087),
    "RElbowYaw":      (-2.0857, 2.0857),
    "RElbowRoll":     ( 0.0087,  1.5620),
    "RWristYaw":      (-1.8239, 1.8239),
    "RHand":          ( 0.0,     1.0),
    "LShoulderPitch": (-2.0857, 2.0857),
    "LShoulderRoll":  ( 0.0087,  1.5620),
    "LElbowYaw":      (-2.0857, 2.0857),
    "LElbowRoll":     (-1.5620, -0.0087),
    "LWristYaw":      (-1.8239, 1.8239),
    "LHand":          ( 0.0,     1.0),
    "HipRoll":        (-0.5149, 0.5149),
    "HipPitch":       (-1.0385, 1.0385),
    "KneePitch":      (-0.5149, 0.5149),
}

def clamp(value, joint_name):
    """Clamp a value to the legal range for a Pepper joint."""
    lo, hi = PEPPER_LIMITS[joint_name]
    return max(lo, min(hi, value))

# ── Geometry Helpers ─────────────────────────────────────────────────────────

def vec3(lm):
    """Extract x, y, z from a landmark as a numpy array."""
    return np.array([lm.x, lm.y, lm.z])

def angle_between(v1, v2):
    """Angle between two vectors in radians."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.acos(np.clip(cos, -1.0, 1.0))

def signed_angle_2d(v1, v2):
    """Signed angle from v1 to v2 in the XY plane."""
    return math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])

def pitch_of(p_from, p_to):
    """Pitch angle: vertical tilt of the vector from p_from to p_to.
       0 = horizontal, positive = downward, negative = upward."""
    d = vec3(p_to) - vec3(p_from)
    horizontal = math.sqrt(d[0]**2 + d[2]**2) + 1e-8
    return math.atan2(d[1], horizontal)

def yaw_of(p_from, p_to):
    """Yaw angle: horizontal rotation of the vector."""
    d = vec3(p_to) - vec3(p_from)
    return math.atan2(d[0], d[2] + 1e-8)

def roll_of(p_from, p_to):
    """Roll: lateral angle in the frontal plane (X-Y)."""
    d = vec3(p_to) - vec3(p_from)
    return math.atan2(d[0], -d[1] + 1e-8)

def elbow_bend_angle(shoulder, elbow, wrist):
    """Calculate elbow flexion angle using 3D vectors."""
    v_upper = vec3(shoulder) - vec3(elbow)
    v_lower = vec3(wrist) - vec3(elbow)
    return math.pi - angle_between(v_upper, v_lower)

def wrist_twist(elbow, wrist, finger):
    """Estimate wrist yaw/pronation by checking how the hand rotates
       relative to the forearm direction."""
    forearm = vec3(wrist) - vec3(elbow)
    hand_dir = vec3(finger) - vec3(wrist)
    # Project hand direction onto plane perpendicular to forearm
    forearm_norm = forearm / (np.linalg.norm(forearm) + 1e-8)
    hand_proj = hand_dir - np.dot(hand_dir, forearm_norm) * forearm_norm
    # Use atan2 on the projected components for twist
    return math.atan2(hand_proj[1], hand_proj[0] + 1e-8)

def hand_openness(wrist, index_tip, pinky_tip):
    """Estimate 0.0 (closed) to 1.0 (open) based on finger spread."""
    spread = np.linalg.norm(vec3(index_tip) - vec3(pinky_tip))
    wrist_to_idx = np.linalg.norm(vec3(index_tip) - vec3(wrist))
    ratio = spread / (wrist_to_idx + 1e-8)
    # Map ratio: ~0.15 => closed, ~0.6 => fully open
    openness = (ratio - 0.15) / (0.6 - 0.15)
    return max(0.0, min(1.0, openness))

# ── Main Extraction Pipeline ────────────────────────────────────────────────

def extract_kinematics(video_path):
    """Process a video file and extract full-body Pepper keyframes."""
    print("  [POSE] Initializing MediaPipe Pose Engine...", file=sys.stderr)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,           # Highest accuracy model
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [ERR] Failed to open video file", file=sys.stderr)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample a keyframe every ~0.3 seconds for smooth motion
    sample_interval = max(1, int(fps * 0.3))
    keyframe_duration = round(sample_interval / fps, 2)

    print(f"  [POSE] Video: {total_frames} frames @ {fps:.1f} fps, sampling every {sample_interval} frames ({keyframe_duration}s)", file=sys.stderr)

    raw_keyframes = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                try:
                    lm = results.pose_landmarks.landmark
                    kf = _extract_single_frame(lm, keyframe_duration)
                    if kf:
                        raw_keyframes.append(kf)
                except Exception as e:
                    print(f"  [WARN] Frame {frame_idx} skipped: {e}", file=sys.stderr)

        frame_idx += 1

    cap.release()
    pose.close()

    # Post-process: deduplicate near-identical consecutive keyframes
    commands = _deduplicate_keyframes(raw_keyframes)

    print(f"  [OK] Compiled {len(commands)} keyframes from {len(raw_keyframes)} raw samples.", file=sys.stderr)
    return commands


def _extract_single_frame(lm, duration):
    """Extract all Pepper joint angles from a single frame's landmarks."""
    joints = []
    angles = []

    # ── HEAD ─────────────────────────────────────────────────────────────
    # HeadYaw: horizontal rotation based on nose relative to shoulder midpoint
    mid_shoulder = type(lm[0])()  # Create a landmark-like object
    mid_shoulder.x = (lm[LM_L_SHOULDER].x + lm[LM_R_SHOULDER].x) / 2
    mid_shoulder.y = (lm[LM_L_SHOULDER].y + lm[LM_R_SHOULDER].y) / 2
    mid_shoulder.z = (lm[LM_L_SHOULDER].z + lm[LM_R_SHOULDER].z) / 2

    head_yaw = yaw_of(mid_shoulder, lm[LM_NOSE]) * 0.8  # Scale down
    head_pitch = pitch_of(mid_shoulder, lm[LM_NOSE]) * 0.8

    joints.extend(["HeadYaw", "HeadPitch"])
    angles.extend([
        round(clamp(head_yaw, "HeadYaw"), 3),
        round(clamp(head_pitch, "HeadPitch"), 3),
    ])

    # ── RIGHT ARM ────────────────────────────────────────────────────────
    r_shoulder_pitch = pitch_of(lm[LM_R_SHOULDER], lm[LM_R_ELBOW])
    r_shoulder_roll  = roll_of(lm[LM_R_SHOULDER], lm[LM_R_ELBOW])
    r_elbow_roll     = elbow_bend_angle(lm[LM_R_SHOULDER], lm[LM_R_ELBOW], lm[LM_R_WRIST])
    r_elbow_yaw      = yaw_of(lm[LM_R_ELBOW], lm[LM_R_WRIST])
    r_wrist_yaw      = wrist_twist(lm[LM_R_ELBOW], lm[LM_R_WRIST], lm[LM_R_INDEX])
    r_hand           = hand_openness(lm[LM_R_WRIST], lm[LM_R_INDEX], lm[LM_R_PINKY])

    joints.extend(["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"])
    angles.extend([
        round(clamp(r_shoulder_pitch, "RShoulderPitch"), 3),
        round(clamp(r_shoulder_roll,  "RShoulderRoll"),  3),
        round(clamp(r_elbow_yaw,      "RElbowYaw"),      3),
        round(clamp(r_elbow_roll,     "RElbowRoll"),     3),
        round(clamp(r_wrist_yaw,      "RWristYaw"),      3),
        round(clamp(r_hand,           "RHand"),          2),
    ])

    # ── LEFT ARM ─────────────────────────────────────────────────────────
    l_shoulder_pitch = pitch_of(lm[LM_L_SHOULDER], lm[LM_L_ELBOW])
    l_shoulder_roll  = -roll_of(lm[LM_L_SHOULDER], lm[LM_L_ELBOW])  # Mirrored
    l_elbow_roll     = -elbow_bend_angle(lm[LM_L_SHOULDER], lm[LM_L_ELBOW], lm[LM_L_WRIST])
    l_elbow_yaw      = yaw_of(lm[LM_L_ELBOW], lm[LM_L_WRIST])
    l_wrist_yaw      = wrist_twist(lm[LM_L_ELBOW], lm[LM_L_WRIST], lm[LM_L_INDEX])
    l_hand           = hand_openness(lm[LM_L_WRIST], lm[LM_L_INDEX], lm[LM_L_PINKY])

    joints.extend(["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw", "LHand"])
    angles.extend([
        round(clamp(l_shoulder_pitch, "LShoulderPitch"), 3),
        round(clamp(l_shoulder_roll,  "LShoulderRoll"),  3),
        round(clamp(l_elbow_yaw,      "LElbowYaw"),      3),
        round(clamp(l_elbow_roll,     "LElbowRoll"),     3),
        round(clamp(l_wrist_yaw,      "LWristYaw"),      3),
        round(clamp(l_hand,           "LHand"),          2),
    ])

    # ── TORSO / HIP ─────────────────────────────────────────────────────
    mid_hip = type(lm[0])()
    mid_hip.x = (lm[LM_L_HIP].x + lm[LM_R_HIP].x) / 2
    mid_hip.y = (lm[LM_L_HIP].y + lm[LM_R_HIP].y) / 2
    mid_hip.z = (lm[LM_L_HIP].z + lm[LM_R_HIP].z) / 2

    hip_roll  = (lm[LM_L_HIP].y - lm[LM_R_HIP].y) * 2.0  # lateral tilt
    hip_pitch = pitch_of(mid_hip, mid_shoulder) + 0.1       # torso lean (offset for upright neutral)

    joints.extend(["HipRoll", "HipPitch"])
    angles.extend([
        round(clamp(hip_roll,  "HipRoll"),  3),
        round(clamp(hip_pitch, "HipPitch"), 3),
    ])

    return {
        "joints": joints,
        "angles": angles,
        "duration": duration,
    }


def _deduplicate_keyframes(keyframes, threshold=0.05):
    """Remove consecutive keyframes that are nearly identical.
       Two keyframes are 'identical' if max angle difference < threshold rad."""
    if len(keyframes) <= 1:
        return keyframes

    filtered = [keyframes[0]]
    for kf in keyframes[1:]:
        prev = filtered[-1]
        max_diff = max(abs(a - b) for a, b in zip(kf["angles"], prev["angles"]))
        if max_diff >= threshold:
            filtered.append(kf)
        else:
            # Extend the duration of the previous keyframe instead
            filtered[-1]["duration"] = round(filtered[-1]["duration"] + kf["duration"], 2)

    return filtered


# ── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mediapipe_engine.py <video_path>", file=sys.stderr)
        print("[]")
        sys.exit(1)

    video_path = sys.argv[1]
    cmds = extract_kinematics(video_path)
    # Output ONLY clean JSON to stdout (server.py parses this)
    print(json.dumps(cmds))
