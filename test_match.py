import sys
sys.path.append("src")

from src.pose_analyzer import PoseAnalyzer

analyzer = PoseAnalyzer()

score = analyzer.analyse_video(
    "reference_videos/test/defense2.avi"
)

score = analyzer.match_against_library(
    score,
    "reference_poses"
)

print("\n=== RESULTS ===")
print("Best Match:", score.best_match)
print("Similarity:", score.overall_similarity)

for name, sim in score.joint_scores.items():
    print(name, sim)