

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import deque




WEIGHTS_PATH = "models/ball_detector.pt"   
CONF_THRESH  = 0.35                        
TRAIL_LENGTH = 30                          
BALL_CLASS   = 0                           

class BallTrackResult:

    def __init__(self):
        self.trajectory: list[tuple[int, int]] = []   # (x, y) per frame
        self.bounce_point: tuple[int, int] | None = None
        self.bounce_frame: int | None = None
        self.annotated_frames: list[np.ndarray] = []
        self.fps: float = 25.0




class BallTracker:
    def __init__(self, weights: str = WEIGHTS_PATH):
        """
        Load YOLOv8n weights.
        Train first with:
            yolo train model=yolov8n.pt data=cricket_ball.yaml epochs=60 imgsz=640
        """
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights}'.\n"
                "Train with: yolo train model=yolov8n.pt data=cricket_ball.yaml epochs=60"
            )
        self.model = YOLO(weights)
        self.trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LENGTH)

    

    def process_video(self, video_path: str) -> BallTrackResult:
        """
        Run detection on every frame of a video.
        Returns a BallTrackResult with annotated frames + trajectory data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        result = BallTrackResult()
        result.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.trail.clear()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            center = self._detect_ball(frame)

            if center:
                self.trail.append(center)
                result.trajectory.append(center)
            else:
               
                result.trajectory.append(None)

            annotated = self._draw_overlay(frame.copy(), center)
            result.annotated_frames.append(annotated)
            frame_idx += 1

        cap.release()

        # Detect bounce after we have the full trajectory
        result.bounce_point, result.bounce_frame = self._find_bounce(
            result.trajectory
        )
        return result

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, tuple | None]:
        """
        Single-frame API — useful for real-time pipelines.
        Returns (annotated_frame, ball_center_or_None).
        """
        center = self._detect_ball(frame)
        if center:
            self.trail.append(center)
        annotated = self._draw_overlay(frame.copy(), center)
        return annotated, center

   
    def _detect_ball(self, frame: np.ndarray) -> tuple[int, int] | None:
        """
        Run YOLOv8 inference and return the highest-confidence ball center.
        Returns None if no detection above threshold.
        """
        results = self.model(frame, verbose=False)[0]

        best_conf = 0.0
        best_center = None

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == BALL_CLASS and conf > CONF_THRESH and conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                best_conf   = conf
                best_center = (cx, cy)

        return best_center

   

    def _find_bounce(
        self, trajectory: list
    ) -> tuple[tuple[int, int] | None, int | None]:
        """
        Bounce = the frame where vertical velocity changes from downward (+y)
        to upward (-y).  We smooth first to reduce noise.

        Returns (bounce_xy, frame_index) or (None, None).
        """
        
        valid = [(i, pt) for i, pt in enumerate(trajectory) if pt is not None]
        if len(valid) < 5:
            return None, None

        indices, points = zip(*valid)
        ys = np.array([p[1] for p in points], dtype=float)

       
        kernel = np.ones(5) / 5
        ys_smooth = np.convolve(ys, kernel, mode="valid")
        offset    = len(ys) - len(ys_smooth)

        
        vy = np.diff(ys_smooth)

       
        for k in range(1, len(vy)):
            if vy[k - 1] > 0 and vy[k] <= 0:
                bounce_valid_idx = k + offset
                orig_idx         = indices[bounce_valid_idx]
                return points[bounce_valid_idx], orig_idx

        return None, None

    
    def _draw_overlay(
        self, frame: np.ndarray, current: tuple[int, int] | None
    ) -> np.ndarray:
        """
        Draw the ball trail (fading white → yellow), current detection circle,
        and bounce marker on the frame.
        """
        trail_list = list(self.trail)
        n = len(trail_list)

      
        for i in range(1, n):
            if trail_list[i - 1] is None or trail_list[i] is None:
                continue
            alpha = i / n                          # 0 = oldest, 1 = newest
            color = (
                int(255 * alpha),                  # B
                int(255 * alpha),                  # G
                int(50  + 200 * alpha),            # R  → yellow tip
            )
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, trail_list[i - 1], trail_list[i], color, thickness)

       
        if current:
            cv2.circle(frame, current, 15, (0,255,255), 3)
            cv2.putText(
    frame,
    "BALL",
    (current[0]+15, current[1]-10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0,255,255),
    2
)
            cv2.circle(frame, current, 3,  (0, 255, 255), -1)  # filled dot

        return frame

    def draw_bounce_marker(
        self, frame: np.ndarray, bounce_pt: tuple[int, int]
    ) -> np.ndarray:
        """Call this separately on the bounce frame to add a ★ marker."""
        cv2.drawMarker(
            frame, bounce_pt, (0, 0, 255),
            markerType=cv2.MARKER_STAR,
            markerSize=20, thickness=2
        )
        cv2.putText(
            frame, "Bounce", (bounce_pt[0] + 12, bounce_pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA
        )
        return frame




def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int = 1):
    """
    Download the cricket ball dataset from Roboflow.
    Dataset: roboflow.com/cricket-2rxrt/cricket-ball-detection
    """
    from roboflow import Roboflow
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location




if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ball_tracker.py <video_path>")
        sys.exit(1)

    tracker = BallTracker()
    result  = tracker.process_video(sys.argv[1])
    # Save annotated video

    if result.annotated_frames:
        h, w = result.annotated_frames[0].shape[:2]

        writer = cv2.VideoWriter(
        "ball_tracking_output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        result.fps,
        (w, h)
    )

    for idx, frame in enumerate(result.annotated_frames):

        if idx == result.bounce_frame and result.bounce_point:
            frame = tracker.draw_bounce_marker(
                frame,
                result.bounce_point
            )

        writer.write(frame)

    writer.release()

    print("Saved: ball_tracking_output.mp4")
    print(result.bounce_point)

    print(f"Tracked {len([p for p in result.trajectory if p])} frames with ball")
    if result.bounce_point:
        print(f"Bounce detected at frame {result.bounce_frame}: {result.bounce_point}")
    else:
        print("No bounce detected")