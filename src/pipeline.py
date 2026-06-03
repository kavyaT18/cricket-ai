
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np




@dataclass
class PipelineResult:
    """
    Everything produced by one pipeline run.
    Fields are None if that feature was skipped or failed.
    """

   
    video_path:      str  = ""
    duration_sec:    float = 0.0

   
    ball_tracked:    bool  = False
    trajectory_pts:  list  = field(default_factory=list)   # list of (x,y)|None
    bounce_point:    tuple | None = None                   # (x,y) pixel
    bounce_frame:    int   | None = None

    
    shot_label:      str   = "unknown"
    shot_confidence: float = 0.0
    shot_top3:       list  = field(default_factory=list)   # [{"shot":..,"confidence":..}]

    
    technique_score:    float = 0.0
    elbow_score:        float = 0.0
    weight_transfer:    float = 0.0
    follow_through:     float = 0.0
    pose_flags:         list  = field(default_factory=list)   # warning strings
    handedness:         str   = "right"

   
    pitch_map_path:     str | None = None
    length_zone:        str = "unknown"    # dominant zone for this delivery
    line_zone:          str = "unknown"
    length_summary:     dict = field(default_factory=dict)
    line_summary:       dict = field(default_factory=dict)

   
    annotated_video_path: str | None = None

   
    elapsed_sec:   float = 0.0
    errors:        list  = field(default_factory=list)   # non-fatal errors
    features_run:  list  = field(default_factory=list)   # which features ran

   

    def to_llm_context(self) -> dict:
        """
        Return a clean dict for the LLM coach.
        Only includes fields that are meaningful for coaching feedback.
        """
        return {
            "shot":            self.shot_label.replace("_", " ").title(),
            "shot_confidence": f"{self.shot_confidence:.0%}",
            "technique_overall": self.technique_score,
            "elbow_position":    self.elbow_score,
            "weight_transfer":   self.weight_transfer,
            "follow_through":    self.follow_through,
            "pose_flags":        self.pose_flags,
            "length_zone":       self.length_zone,
            "line_zone":         self.line_zone,
            "handedness":        self.handedness,
            "bounce_detected":   self.bounce_point is not None,
            "context_best_match": self.pose_flags[0]
        }

    def summary(self) -> str:
        """One-line human readable summary."""
        parts = [f"Shot={self.shot_label}({self.shot_confidence:.0%})"]
        if self.technique_score:
            parts.append(f"Technique={self.technique_score}/100")
        if self.length_zone != "unknown":
            parts.append(f"Length={self.length_zone}")
        if self.bounce_point:
            parts.append(f"Bounce@frame{self.bounce_frame}")
        parts.append(f"[{self.elapsed_sec:.1f}s]")
        return "  |  ".join(parts)




class Pipeline:
    

    def __init__(
        self,
       
        run_ball_tracker:   bool = True,
        run_shot_classifier: bool = False,
        run_pose_analyzer:  bool = True,
        run_pitch_map:      bool = True,
       
        ball_weights:  str = "models/ball_detector.pt",
        shot_weights:  str = "models/shot_classifier.pt",
        output_dir:    str = "output",
       
        handedness:    str = "right",
       
        pitch_map_title: str = "Delivery Analysis",
    ):
        self.flags = {
            "ball":  run_ball_tracker,
            "shot":  run_shot_classifier,
            "pose":  run_pose_analyzer,
            "pitch": run_pitch_map,
        }
        self.ball_weights    = ball_weights
        self.shot_weights    = shot_weights
        self.output_dir      = Path(output_dir)
        self.handedness      = handedness
        self.pitch_map_title = pitch_map_title

        self.output_dir.mkdir(parents=True, exist_ok=True)

       
        self._tracker    = None
        self._classifier = None
        self._analyzer   = None

   

    def run(self, video_path: str) -> PipelineResult:
        """
        Run the full pipeline on one video.
        Returns a PipelineResult with everything filled in.
        """
        t0 = time.time()
        result = PipelineResult(video_path=video_path, handedness=self.handedness)

        if not Path(video_path).exists():
            result.errors.append(f"Video not found: {video_path}")
            return result

        result.duration_sec = _video_duration(video_path)
        video_stem = Path(video_path).stem

        print(f"\n{'─'*50}")
        print(f"  Cricket Pipeline  →  {Path(video_path).name}")
        print(f"{'─'*50}")

        
        ball_result = None
        if self.flags["ball"]:
            ball_result = self._run_ball_tracker(video_path, result)

       
        if self.flags["shot"]:
            self._run_shot_classifier(video_path, result)

        
        pose_result = None
        if self.flags["pose"]:
            pose_result = self._run_pose_analyzer(video_path, result)

        
        if self.flags["pitch"]:
            self._run_pitch_map(
                ball_result, result,
                save_path=str(self.output_dir / f"{video_stem}_pitch_map.png"),
            )

        
        out_video = str(self.output_dir / f"{video_stem}_annotated.mp4")
        self._compose_output(
            video_path, ball_result, pose_result, result, out_video
        )
        result.annotated_video_path = out_video

        result.elapsed_sec = round(time.time() - t0, 2)
        print(f"\n  ✓ Done in {result.elapsed_sec}s")
        print(f"  {result.summary()}")
        return result

    

    def _run_ball_tracker(self, video_path: str, result: PipelineResult):
        """Run Feature 1. Returns raw BallTrackResult or None on failure."""
        print("\n[1/4] Ball tracker...")
        try:
            from src.ball_tracker import BallTracker
            if self._tracker is None:
                self._tracker = BallTracker(weights=self.ball_weights)

            track = self._tracker.process_video(video_path)
            result.ball_tracked   = True
            result.trajectory_pts = track.trajectory
            result.bounce_point   = track.bounce_point
            result.bounce_frame   = track.bounce_frame
            result.features_run.append("ball_tracker")

            n_detected = sum(1 for p in track.trajectory if p is not None)
            print(f"     Ball detected in {n_detected}/{len(track.trajectory)} frames")
            if track.bounce_point:
                print(f"     Bounce @ frame {track.bounce_frame}  {track.bounce_point}")
            return track

        except FileNotFoundError as e:
            msg = f"Ball tracker skipped: {e}"
            print(f"     [SKIP] {msg}")
            result.errors.append(msg)
            return None
        except Exception as e:
            msg = f"Ball tracker error: {e}"
            print(f"     [ERROR] {msg}")
            result.errors.append(msg)
            traceback.print_exc()
            return None

    def _run_shot_classifier(self, video_path: str, result: PipelineResult):
        """Run Feature 2."""
        print("\n[2/4] Shot classifier...")
        try:
            from src.shot_classifier import ShotInference
            if self._classifier is None:
                self._classifier = ShotInference(weights=self.shot_weights)

            pred = self._classifier.classify_video(video_path)
            result.shot_label      = pred["shot"]
            result.shot_confidence = pred["confidence"]
            result.shot_top3       = self._classifier.top_k(video_path, k=3)
            result.features_run.append("shot_classifier")

            print(f"     Shot: {result.shot_label}  ({result.shot_confidence:.0%})")

        except FileNotFoundError as e:
            msg = f"Shot classifier skipped: {e}"
            print(f"     [SKIP] {msg}")
            result.errors.append(msg)
        except Exception as e:
            msg = f"Shot classifier error: {e}"
            print(f"     [ERROR] {msg}")
            result.errors.append(msg)
            traceback.print_exc()

    def _run_pose_analyzer(self, video_path: str, result: PipelineResult):
        """Run Feature 3. Returns raw TechniqueScore or None."""
        print("\n[3/4] Pose analyzer...")
        try:
            from src.pose_analyzer import PoseAnalyzer
            if self._analyzer is None:
                self._analyzer = PoseAnalyzer(handedness=self.handedness)

            score = self._analyzer.analyse_video(video_path)
            
            result.technique_score = score.overall_similarity
            result.pose_flags = [
    f"Best Match: {score.best_match}"
]

            print(f"     Technique: {result.technique_score}")
            
            return score

        except Exception as e:
            msg = f"Pose analyzer error: {e}"
            print(f"     [ERROR] {msg}")
            result.errors.append(msg)
            traceback.print_exc()
            return None

    def _run_pitch_map(self, ball_result, result: PipelineResult, save_path: str):
        """Run Feature 4. Uses ball_result if available, else demo data."""
        print("\n[4/4] Pitch map...")
        try:
            from src.pitch_map import PitchMap, build_from_ball_tracker, build_demo_map

            if ball_result and ball_result.bounce_point:
                pm = build_from_ball_tracker(ball_result, shot_label=result.shot_label)
            else:
                # No real bounce data — use a single synthetic delivery
                # so the pitch map still renders for the dashboard
                print("     No bounce point from tracker — using demo delivery")
                pm = build_demo_map(n=1, seed=hash(result.video_path) % 9999)

            map_result = pm.render(
                save_path=save_path,
                title=f"{self.pitch_map_title}  —  {result.shot_label.replace('_', ' ').title()}",
            )

            result.pitch_map_path   = save_path
            result.length_summary   = map_result.length_summary
            result.line_summary     = map_result.line_summary
            result.features_run.append("pitch_map")

            # Dominant zones
            if map_result.length_summary:
                result.length_zone = max(
                    map_result.length_summary, key=map_result.length_summary.get
                )
            if map_result.line_summary:
                result.line_zone = max(
                    map_result.line_summary, key=map_result.line_summary.get
                )

            print(f"     Saved → {save_path}")

        except Exception as e:
            msg = f"Pitch map error: {e}"
            print(f"     [ERROR] {msg}")
            result.errors.append(msg)
            traceback.print_exc()

    # ── Output video composition ──────────────────────────────────────────────

    def _compose_output(
        self,
        video_path:  str,
        ball_result,
        pose_result,
        pipeline_result: PipelineResult,
        out_path:    str,
    ):
        """
        Merge ball trajectory + pose skeleton + HUD into one output video.
        Picks the richest annotated frame source available.
        """
        print(f"\n[out] Composing annotated video → {out_path}")
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Choose best frame source: pose (has skeleton) > ball > raw
            if pose_result and pose_result.annotated_frames:
                base_frames = pose_result.annotated_frames
            elif ball_result and ball_result.annotated_frames:
                base_frames = ball_result.annotated_frames
            else:
                base_frames = _read_raw_frames(video_path)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

            for idx, frame in enumerate(base_frames):
                # Blend ball trajectory overlay on top of pose skeleton
                if (ball_result and ball_result.annotated_frames
                        and idx < len(ball_result.annotated_frames)):
                    ball_frame = ball_result.annotated_frames[idx]
                    cv2.addWeighted(ball_frame, 0.35, frame, 0.65, 0, frame)

                # Bounce marker on the exact bounce frame ±3 frames
                if (ball_result and ball_result.bounce_point
                        and ball_result.bounce_frame is not None
                        and abs(idx - ball_result.bounce_frame) <= 3):
                    _draw_bounce_marker(frame, ball_result.bounce_point)

                # HUD: shot label + technique scores
                frame = _draw_hud(frame, pipeline_result, idx)
                writer.write(frame)

            writer.release()

        except Exception as e:
            print(f"     [ERROR] Output video failed: {e}")
            pipeline_result.errors.append(f"Output video: {e}")

    # ── Convenience: run on multiple clips (e.g. full over) ───────────────────

    def run_batch(self, video_paths: list[str]) -> list[PipelineResult]:
        """
        Run the pipeline on a list of videos.
        Useful for analysing a full over (6 deliveries).
        Returns list of PipelineResults in the same order.
        """
        results = []
        for i, vp in enumerate(video_paths, 1):
            print(f"\n{'='*50}")
            print(f"  Video {i}/{len(video_paths)}: {Path(vp).name}")
            results.append(self.run(vp))
        return results




def _draw_hud(frame: np.ndarray, r: PipelineResult, frame_idx: int) -> np.ndarray:
    """Top-right overlay panel with shot + technique scores."""
    h, w = frame.shape[:2]
    pw, ph = 255, 145
    x0, y0 = w - pw - 10, 10

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    white = (235, 235, 235)
    gold  = (0, 210, 255)
    green = (80, 220, 80)
    yc    = y0 + 22

    # Shot label
    shot_display = r.shot_label.replace("_", " ").title()
    cv2.putText(frame, f"{shot_display}", (x0 + 8, yc),
                font, 0.52, gold, 1, cv2.LINE_AA)
    yc += 18
    cv2.putText(frame, f"Conf: {r.shot_confidence:.0%}", (x0 + 8, yc),
                font, 0.42, white, 1, cv2.LINE_AA)
    yc += 22

    # Technique scores
    if r.technique_score:
        cv2.putText(frame, f"Technique: {r.technique_score:.0f}/100", (x0 + 8, yc),
                    font, 0.48, green, 1, cv2.LINE_AA)
        yc += 18
        cv2.putText(frame, f"Elbow:{r.elbow_score:.0f}  Wt:{r.weight_transfer:.0f}  FT:{r.follow_through:.0f}",
                    (x0 + 8, yc), font, 0.38, white, 1, cv2.LINE_AA)
        yc += 18

    # Length zone
    if r.length_zone != "unknown":
        cv2.putText(frame, f"Length: {r.length_zone}", (x0 + 8, yc),
                    font, 0.40, white, 1, cv2.LINE_AA)

    return frame


def _draw_bounce_marker(frame: np.ndarray, pt: tuple):
    cv2.drawMarker(frame, pt, (0, 60, 255),
                   markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)
    cv2.putText(frame, "Bounce", (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1, cv2.LINE_AA)


def _video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return round(frames / fps, 2)


def _read_raw_frames(path: str) -> list:
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames



if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Cricket Analytics Pipeline")
    parser.add_argument("video",               help="Path to batting video")
    parser.add_argument("--hand",              default="right", choices=["right","left"])
    parser.add_argument("--skip-ball",         action="store_true")
    parser.add_argument("--skip-shot",         action="store_true")
    parser.add_argument("--skip-pose",         action="store_true")
    parser.add_argument("--output-dir",        default="output")
    args = parser.parse_args()

    pipe = Pipeline(
        run_ball_tracker    = not args.skip_ball,
        run_shot_classifier = not args.skip_shot,
        run_pose_analyzer   = not args.skip_pose,
        handedness          = args.hand,
        output_dir          = args.output_dir,
    )
    result = pipe.run(args.video)

    # Print LLM context so you can see what gets sent to Groq
    print("\n── LLM Context (sent to Groq) ──")
    print(json.dumps(result.to_llm_context(), indent=2))

    if result.errors:
        print(f"\n── Non-fatal errors ({len(result.errors)}) ──")
        for e in result.errors:
            print(f"  • {e}")