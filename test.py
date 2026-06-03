from src.pose_analyzer import PoseAnalyzer

analyzer = PoseAnalyzer()

analyzer.save_reference(
    "reference_videos/latecut.avi",
    "reference_poses/latecut.npy"
)

analyzer.save_reference(
    "reference_videos/pull_shot.avi",
    "reference_poses/pull_shot.npy"
)

analyzer.save_reference(
    "reference_videos/scoop.avi",
    "reference_poses/scoop.npy"
)

analyzer.save_reference(
    "reference_videos/straightdrive.avi",
    "reference_poses/straightdrive.npy"
)
analyzer.save_reference(
    "reference_videos/uppercut.avi",
    "reference_poses/uppercut.npy"
)
analyzer.save_reference(
    "reference_videos/test/defense.avi",
    "reference_poses/defense.npy"
)