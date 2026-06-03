
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ball_tracker    import BallTracker
from src.shot_classifier import ShotInference
from src.pose_analyzer   import PoseAnalyzer




def parse_args():
    parser = argparse.ArgumentParser(description="Cricket AI — Phase 2 runner")
    parser.add_argument("--video",      required=True,          help="Input video path")
    parser.add_argument("--output",     default="output/phase2_result.mp4")
    parser.add_argument("--hand",       default="right",        choices=["right", "left"])
    parser.add_argument("--skip-ball",  action="store_true",    help="Skip ball tracker (no model weights yet)")
    parser.add_argument("--skip-shot",  action="store_true",    help="Skip shot classifier")
    parser.add_argument("--skip-pose",  action="store_true",    help="Skip pose analyzer")
    return parser.parse_args()




def draw_hud(
    frame: np.ndarray,
    shot_result: dict | None,
    pose_score,
    bounce_frame: int | None,
    frame_idx: int,
) -> np.ndarray:
    """
    Draw a heads-up display panel in the top-right corner showing
    shot label, confidence, and technique scores.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 140
    x0, y0 = w - panel_w - 10, 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    white  = (240, 240, 240)
    yellow = (0, 220, 255)
    green  = (80, 220, 80)
    y_cur  = y0 + 22

   
    if shot_result:
        shot_label = shot_result["shot"].replace("_", " ").title()
        conf       = shot_result["confidence"]
        cv2.putText(frame, f"Shot: {shot_label}", (x0 + 8, y_cur),
                    font, 0.50, yellow, 1, cv2.LINE_AA)
        y_cur += 20
        cv2.putText(frame, f"Conf: {conf:.0%}", (x0 + 8, y_cur),
                    font, 0.45, white, 1, cv2.LINE_AA)
        y_cur += 24

  
    if pose_score and pose_score.overall > 0:
        cv2.putText(frame, f"Technique: {pose_score.overall:.0f}/100", (x0 + 8, y_cur),
                    font, 0.50, green, 1, cv2.LINE_AA)
        y_cur += 20
        cv2.putText(frame, f"Elbow : {pose_score.elbow_position:.0f}/100", (x0 + 8, y_cur),
                    font, 0.40, white, 1, cv2.LINE_AA)
        y_cur += 18
        cv2.putText(frame, f"Wt Xfr: {pose_score.weight_transfer:.0f}/100", (x0 + 8, y_cur),
                    font, 0.40, white, 1, cv2.LINE_AA)
        y_cur += 18
        cv2.putText(frame, f"Follow: {pose_score.follow_through:.0f}/100", (x0 + 8, y_cur),
                    font, 0.40, white, 1, cv2.LINE_AA)

    
    if bounce_frame is not None and abs(frame_idx - bounce_frame) < 10:
        cv2.putText(frame, "★ BOUNCE", (x0 + 8, y0 + panel_h - 8),
                    font, 0.55, (0, 80, 255), 2, cv2.LINE_AA)

    return frame




def main():
    args = parse_args()
    video_path = args.video

    if not Path(video_path).exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

   
    ball_result  = None
    if not args.skip_ball:
        
        try:
            tracker     = BallTracker()
            ball_result = tracker.process_video(video_path)
            print(f"      Detected ball in {len([p for p in ball_result.trajectory if p])} frames")
            if ball_result.bounce_point:
                print(f"      Bounce at frame {ball_result.bounce_frame}: {ball_result.bounce_point}")
        except FileNotFoundError as e:
            print(f"      [SKIP] {e}")
            ball_result = None
    else:
        print("[1/3] Ball tracker skipped (--skip-ball)")

   
    shot_result = None
    if not args.skip_shot:
       
        try:
            classifier  = ShotInference()
            shot_result = classifier.classify_video(video_path)
            print(f"      Shot: {shot_result['shot']}  ({shot_result['confidence']:.0%} confidence)")
        except FileNotFoundError as e:
            print(f"      [SKIP] {e}")
    else:
        print("[2/3] Shot classifier skipped (--skip-shot)")

   
    pose_score = None
    if not args.skip_pose:
        print("\n[3/3] Running pose analyzer (MediaPipe)...")
        analyzer   = PoseAnalyzer(handedness=args.hand)
        pose_score = analyzer.analyse_video(video_path)
        print(f"      Overall technique: {pose_score.overall}/100")
        for flag in pose_score.flags:
            print(f"      ⚠  {flag}")
    else:
        print("[3/3] Pose analyzer skipped (--skip-pose)")

   
    print(f"\nComposing annotated output → {args.output}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

   
    if pose_score and pose_score.annotated_frames:
        source_frames = pose_score.annotated_frames
    elif ball_result and ball_result.annotated_frames:
        source_frames = ball_result.annotated_frames
    else:
        # Fall back to raw frames
        cap = cv2.VideoCapture(video_path)
        source_frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            source_frames.append(f)
        cap.release()

    for idx, frame in enumerate(source_frames):
       
        if ball_result and ball_result.annotated_frames and idx < len(ball_result.annotated_frames):
            ball_frame = ball_result.annotated_frames[idx]
           
            cv2.addWeighted(ball_frame, 0.4, frame, 0.6, 0, frame)

      
        if (ball_result and ball_result.bounce_frame == idx
                and ball_result.bounce_point):
            from src.ball_tracker import BallTracker as BT
            frame = BT.__new__(BT)
            from src.ball_tracker import BallTracker
            frame = BallTracker.draw_bounce_marker(
                BallTracker.__new__(BallTracker), frame, ball_result.bounce_point
            )

      
        frame = draw_hud(frame, shot_result, pose_score, 
                         ball_result.bounce_frame if ball_result else None,
                         idx)
        writer.write(frame)

    writer.release()
   
    if shot_result:
        print(f"  Shot classified : {shot_result['shot']} ({shot_result['confidence']:.0%})")
    if pose_score:
        print(f"  Elbow position  : {pose_score.elbow_position}/100")
        print(f"  Weight transfer : {pose_score.weight_transfer}/100")
        print(f"  Follow-through  : {pose_score.follow_through}/100")
        print(f"  Overall         : {pose_score.overall}/100")
    if ball_result and ball_result.bounce_point:
        print(f"  Bounce detected : frame {ball_result.bounce_frame} @ {ball_result.bounce_point}")
    print()


if __name__ == "__main__":
    main()