

from __future__ import annotations

import cv2
import math
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial.distance import cosine
from fastdtw import fastdtw


LM = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_heel":      29, "right_heel":     30,
}


IDEAL_ELBOW_MIN = 100
IDEAL_ELBOW_MAX = 150

# Ideal follow-through angle (degrees above horizontal)
IDEAL_FOLLOW_THROUGH_MIN = 30
IDEAL_FOLLOW_THROUGH_MAX = 80




@dataclass
class FramePose:
    """Pose data for a single frame."""
    frame_idx:   int
    landmarks:   dict[str, tuple[float, float, float]]  # name → (x, y, z) normalised
    elbow_angle: float | None = None       # leading arm elbow (degrees)
    hip_shift:   float | None = None       # weight transfer metric (hip x-diff)
    wrist_height: float | None = None      # follow-through height metric
    pose_vector: np.ndarray | None = None


@dataclass
class TechniqueScore:
    overall_similarity: float = 0.0
    best_match: str = "unknown"

    frame_poses: list[FramePose] = field(default_factory=list)
    annotated_frames: list[np.ndarray] = field(default_factory=list)

    joint_scores: dict = field(default_factory=dict)



class PoseAnalyzer:
   
    def __init__(self, handedness: str = "right"):
        
        self.mp_pose    = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose       = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,       # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Leading arm = the arm that points toward the bowler
        self.leading_side = "left" if handedness == "right" else "right"
        self.trail_side   = "right" if handedness == "right" else "left"


    def _pose_vector(self, lms):

        vec = []

        for _, p in lms.items():
            vec.extend([
            p[0],
            p[1],
            p[2]
        ])

        return np.array(vec, dtype=np.float32)

   
    def analyse_video(self, video_path: str) -> TechniqueScore:
       
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        score  = TechniqueScore()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_pose, annotated = self._process_frame(frame, frame_idx)
            if frame_pose:
                score.frame_poses.append(frame_pose)
            score.annotated_frames.append(annotated)
            frame_idx += 1

        cap.release()
        score = self.match_against_library(score)
        return score

    def analyse_frame(self, frame: np.ndarray, frame_idx: int = 0) -> tuple[FramePose | None, np.ndarray]:
        """Single-frame API."""
        return self._process_frame(frame, frame_idx)

   

    def _process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> tuple[FramePose | None, np.ndarray]:
        """
        Run MediaPipe on one frame.
        Returns (FramePose, annotated_frame).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        annotated = frame.copy()

        if not results.pose_landmarks:
            return None, annotated

        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(255, 255, 0), thickness=2
            ),
        )

        # Extract normalised landmark dict
        h, w = frame.shape[:2]
        lms = {}
        for name, idx in LM.items():
            lm = results.pose_landmarks.landmark[idx]
            lms[name] = (lm.x, lm.y, lm.z)   # normalised 0–1

        fp = FramePose(frame_idx=frame_idx, landmarks=lms)

        # Compute per-frame metrics
       
        fp.pose_vector = self._pose_vector(lms)

        # Overlay numeric metrics on frame
        self._draw_metrics(annotated, fp, w, h)

        return fp, annotated

   

    def _draw_metrics(self, frame: np.ndarray, fp: FramePose, w: int, h: int):
        """Overlay elbow angle and follow-through angle on the frame."""
        y = 30
        if fp.elbow_angle is not None:
            cv2.putText(frame, f"Elbow: {fp.elbow_angle:.0f}deg",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y += 28
        if fp.wrist_height is not None:
            cv2.putText(frame, f"Follow-thru: {fp.wrist_height:.0f}deg",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    def __del__(self):
        if hasattr(self, "pose"):
            self.pose.close()
    

    def compare_to_reference(
    self,
    score,
    reference_path
):
        ref = np.load(reference_path)

        user = np.array([
        fp.pose_vector
        for fp in score.frame_poses
        if fp.pose_vector is not None
    ])

        if len(user) < 5:
            return 0.0

        distance, _ = fastdtw(
        user,
        ref,
        dist=cosine
    )

        similarity = max(
    0,
    100 * np.exp(-distance)
)
        print("distance =", distance)

        return round(similarity, 2)

    def match_against_library(
    self,
    score,
    ref_dir="reference_poses"
):
       
        results = {}

        for ref in Path(ref_dir).glob("*.npy"):

            sim = self.compare_to_reference(
            score,
            str(ref)
        )

            results[ref.stem] = sim

        best = max(
        results,
        key=results.get
    )

        score.best_match = best
        score.overall_similarity = results[best]
        score.joint_scores = results

        return score


    def _angle_3pts(a, b, c) -> float:
        """
    Angle at vertex b, given three (x, y, z) normalised points.
    Returns degrees in [0, 180].
    """
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(math.degrees(math.acos(cosine)))


    def save_reference(self, video_path, output_path):

        score = self.analyse_video(video_path)

        seq = np.array([
        fp.pose_vector
        for fp in score.frame_poses
        if fp.pose_vector is not None
    ])

        np.save(output_path, seq)

        print(f"Saved reference: {output_path}")



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pose_analyzer.py <video_path> [right|left]")
        sys.exit(1)

    vid    = sys.argv[1]
    hand   = sys.argv[2] if len(sys.argv) > 2 else "right"
    analyzer = PoseAnalyzer(handedness=hand)
    score    = analyzer.analyse_video(vid)

    print("\n=== Pose Matching ===")

    print(
    f"Best Match: {score.best_match}"
)

    print(
    f"Similarity: {score.overall_similarity:.2f}%"
)

    print("\nAll References:")

    for name, sim in score.joint_scores.items():
        print(
        f"{name:<25} {sim:.2f}%"
    )
    