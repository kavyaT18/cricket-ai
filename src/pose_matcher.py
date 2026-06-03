from __future__ import annotations

import numpy as np
import mediapipe as mp
import cv2
from pathlib import Path
from scipy.spatial.distance import cosine
from fastdtw import fastdtw


class PoseMatcher:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

  

    def extract_pose_sequence(self, video_path: str):
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(video_path)

        sequence = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:

                frame_vec = []

                for lm in result.pose_landmarks.landmark:
                    frame_vec.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                sequence.append(frame_vec)

        cap.release()
        pose.close()

        return np.array(sequence, dtype=np.float32)

   

    def normalize_sequence(self, seq):

        normalized = []

        for frame in seq:

            frame = frame.reshape(33, 3)

            left_hip = frame[23]
            right_hip = frame[24]

            hip_center = (left_hip + right_hip) / 2

            left_shoulder = frame[11]
            right_shoulder = frame[12]

            shoulder_width = np.linalg.norm(
                left_shoulder - right_shoulder
            )

            shoulder_width = max(
                shoulder_width,
                1e-6
            )

            frame = frame - hip_center
            frame = frame / shoulder_width

            normalized.append(
                frame.flatten()
            )

        return np.array(normalized)

   

    def compare_sequences(
        self,
        user_seq,
        ref_seq
    ):

        distance, _ = fastdtw(
            user_seq,
            ref_seq,
            dist=cosine
        )

        similarity = max(
            0,
            100 * np.exp(-distance / 100)
        )

        return round(similarity, 2)

   

    def create_reference(
        self,
        video_path,
        output_path
    ):

        seq = self.extract_pose_sequence(
            video_path
        )

        seq = self.normalize_sequence(seq)

        np.save(
            output_path,
            seq
        )

        print(
            f"Saved reference: {output_path}"
        )

  
    def match_video(
        self,
        user_video,
        reference_dir
    ):

        user_seq = self.extract_pose_sequence(
            user_video
        )

        user_seq = self.normalize_sequence(
            user_seq
        )

        results = []

        for ref_file in Path(reference_dir).glob("*.npy"):

            ref_seq = np.load(ref_file)

            score = self.compare_sequences(
                user_seq,
                ref_seq
            )

            results.append({
                "reference": ref_file.stem,
                "score": score
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--create-reference",
        type=str
    )

    parser.add_argument(
        "--output",
        type=str
    )

    parser.add_argument(
        "--match",
        type=str
    )

    parser.add_argument(
        "--refs",
        type=str,
        default="reference_poses"
    )

    args = parser.parse_args()

    matcher = PoseMatcher()

    if args.create_reference:

        matcher.create_reference(
            args.create_reference,
            args.output
        )

    elif args.match:

        results = matcher.match_video(
            args.match,
            args.refs
        )

        print("\n=== Pose Matching Results ===\n")

        for r in results:
            print(
                f"{r['reference']}: {r['score']:.2f}%"
            )

        if results:
            print(
                f"\nBest Match: {results[0]['reference']}"
            )