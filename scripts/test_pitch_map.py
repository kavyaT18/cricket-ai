import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.ball_tracker import BallTracker
from src.pitch_map import PitchMap

tracker = BallTracker()

result = tracker.process_video("reference_videos/uppercut.avi")

print("Bounce:", result.bounce_point)

pm = PitchMap()

if result.bounce_point:

    x, y = result.bounce_point

    pm.from_pixel_bounce(
        x,
        y,
        frame_w=1280,
        frame_h=720
    )

pm.render(
    save_path="output/real_pitch_map.png",
    title="Single Delivery"
)