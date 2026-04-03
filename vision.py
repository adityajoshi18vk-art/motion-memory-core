import cv2
import time
import sys

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_FILE     = "demo_motion.mp4"
RECORD_DURATION = 5          # seconds
WARMUP_DURATION = 2          # seconds for "Get Ready" warning
FONT            = cv2.FONT_HERSHEY_DUPLEX
OVERLAY_COLOR   = (0, 0, 255)   # Red  (BGR)
SHADOW_COLOR    = (0, 0, 100)   # Dark-red shadow for depth


def get_ready_countdown(duration: int) -> None:
    """Print a blocking 'Get Ready' countdown in the terminal."""
    print("\n" + "=" * 45)
    print("  [CAMERA]  Motion-Memory Recorder  [CAMERA]")
    print("=" * 45)
    for remaining in range(duration, 0, -1):
        sys.stdout.write(f"\r  ... Get Ready... recording starts in {remaining}s ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r  REC > Recording NOW!                         \n")
    sys.stdout.flush()
    print("=" * 45 + "\n")


def draw_recording_overlay(frame, elapsed: float, total: float) -> None:
    """
    Burn a professional-looking RECORDING badge + countdown timer
    directly onto the given frame (in-place).
    """
    h, w = frame.shape[:2]
    remaining = max(0.0, total - elapsed)

    # ── "● RECORDING" label ──────────────────────────────────────────────────
    rec_text  = "RECORDING"
    rec_scale = 1.1
    rec_thick = 2

    # Shadow / outline pass
    cv2.putText(frame, rec_text, (28, 48),
                FONT, rec_scale, SHADOW_COLOR, rec_thick + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, rec_text, (28, 48),
                FONT, rec_scale, OVERLAY_COLOR, rec_thick, cv2.LINE_AA)

    # Blinking red dot (visible roughly every even second)
    dot_visible = int(elapsed) % 2 == 0
    if dot_visible:
        cv2.circle(frame, (14, 38), 8, OVERLAY_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (14, 38), 9, SHADOW_COLOR,  1,  cv2.LINE_AA)

    # ── Countdown timer ──────────────────────────────────────────────────────
    timer_text  = f"{remaining:.1f}s"
    timer_scale = 0.9
    timer_thick = 2

    (tw, th), _ = cv2.getTextSize(timer_text, FONT, timer_scale, timer_thick)
    tx = w - tw - 20
    ty = 45

    # Shadow pass
    cv2.putText(frame, timer_text, (tx + 2, ty + 2),
                FONT, timer_scale, SHADOW_COLOR, timer_thick + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, timer_text, (tx, ty),
                FONT, timer_scale, OVERLAY_COLOR, timer_thick, cv2.LINE_AA)

    # ── Progress bar at the bottom ───────────────────────────────────────────
    bar_h  = 6
    bar_y  = h - bar_h - 4
    bar_x0 = 10
    bar_x1 = w - 10
    bar_full_w = bar_x1 - bar_x0

    # Background track
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h),
                  (40, 40, 40), -1, cv2.LINE_AA)

    # Fill proportional to elapsed time
    progress_w = int(bar_full_w * (elapsed / total))
    if progress_w > 0:
        cv2.rectangle(frame,
                      (bar_x0, bar_y),
                      (bar_x0 + progress_w, bar_y + bar_h),
                      OVERLAY_COLOR, -1, cv2.LINE_AA)


def main() -> None:
    # ── 1. Terminal "Get Ready" warning ──────────────────────────────────────
    get_ready_countdown(WARMUP_DURATION)

    # ── 2. Open webcam ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the default webcam. Exiting.")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Camera   : {width}×{height} @ {fps:.1f} fps")
    print(f"  Output   : {OUTPUT_FILE}")
    print(f"  Duration : {RECORD_DURATION}s\n")

    # ── 3. Set up video writer ────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))

    # ── 4. Record for exactly RECORD_DURATION seconds ────────────────────────
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame, stopping early.")
            break

        elapsed = time.time() - start_time
        if elapsed >= RECORD_DURATION:
            break

        # Burn overlay onto the frame
        draw_recording_overlay(frame, elapsed, RECORD_DURATION)

        # Write to file
        writer.write(frame)

        # Live preview window
        cv2.imshow("Motion-Memory Recorder  [press Q to abort]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n  [Aborted by user]")
            break

    # ── 5. Clean up ──────────────────────────────────────────────────────────
    actual_duration = time.time() - start_time
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"  OK > Recording complete — {min(actual_duration, RECORD_DURATION):.2f}s saved to '{OUTPUT_FILE}'")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
