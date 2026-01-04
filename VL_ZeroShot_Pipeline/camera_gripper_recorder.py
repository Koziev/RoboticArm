import cv2
import threading
import time
import numpy as np
import datetime
import pathlib
import os
import json
import PIL
from PIL import Image


# ----------------------------
# Shared state
# ----------------------------
latest_frame = None
processed_frame = None

frame_lock = threading.Lock()
proc_lock = threading.Lock()

running = True


# ----------------------------
# Camera capture thread
# ----------------------------
def camera_thread():
    global latest_frame

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            latest_frame = frame

    cap.release()



class GripperTracker:
    def __init__(self, hsv_lower, hsv_upper, min_area=150):
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_area

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area:
            return None, mask

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return {
            "center": (cx, cy),
            "area": area
        }, mask


class VisionSystem:
    def __init__(self, gripper_tracker):
        #self.ball_detector = ball_detector
        self.gripper_tracker = gripper_tracker

    def process(self, frame):
        #ball, ball_mask = self.ball_detector.detect(frame)
        grip, grip_mask = self.gripper_tracker.detect(frame)

        return {
            #"ball": ball,
            "gripper": grip,
            #"ball_mask": ball_mask,
            "gripper_mask": grip_mask
        }



# ----------------------------
# Main pipeline thread
# (stub: pretend this is OpenVLA + IK)
# ----------------------------
def pipeline_thread():
    global processed_frame

    # Blue marker on gripper (example)
    gripper_tracker = GripperTracker(
        hsv_lower=[100, 150, 50],
        hsv_upper=[140, 255, 255]
    )
    vision = VisionSystem(gripper_tracker)

    frame_count = 0
    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        vresult = vision.process(frame)
        if vresult["gripper"]:
            c = vresult["gripper"]["center"]
            cv2.circle(frame, c, 8, (255, 0, 0), -1)
            cv2.putText(frame, "GRIPPER", c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        with proc_lock:
            processed_frame = frame

        if vresult["gripper"] is not None:
            frame_count += 1
            ts = datetime.datetime.now().strftime("%I:%M:%S on %d.%m.%Y")
            frame_fp = os.path.join(output_dir, f"frame_{ts}.png")
            im1 = PIL.Image.fromarray(frame)
            im1.save(frame_fp)

            with open(os.path.join(output_dir, f"frame {ts}.json"), "w") as f:
                data = {"frame_fp": frame_fp,
                        "timestamp_ns": time.time_ns(),
                        "timestamp": ts,
                        "gripper": vresult["gripper"]
                        }
                json.dump(data, f, indent=4)

        time.sleep(0.1)

# ----------------------------
# OpenCV visualization thread
# ----------------------------
def display_thread():
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 960)

    while running:
        with proc_lock:
            if processed_frame is None:
                continue
            frame = processed_frame.copy()

        cv2.imshow("Camera", frame)

        # THIS is what keeps the window fresh
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    output_dir = "./camera_frames"
    pathlib.Path(output_dir).mkdir(exist_ok=True)

    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_pipe = threading.Thread(target=pipeline_thread, daemon=True)
    t_disp = threading.Thread(target=display_thread, daemon=True)

    t_cam.start()
    t_pipe.start()
    t_disp.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        running = False
        time.sleep(0.5)
