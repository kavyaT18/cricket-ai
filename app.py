import streamlit as st
import tempfile
from pathlib import Path

from src.pipeline import Pipeline
from src.llm_coach import LLMCoach

st.set_page_config(
page_title="Cricket Analytics",
page_icon="🏏",
layout="wide"
)

st.title("🏏 Cricket Analytics Dashboard")

uploaded_file = st.file_uploader(
"Upload Batting Video",
type=["mp4", "avi", "mov"]
)

if uploaded_file:


    with tempfile.NamedTemporaryFile(delete=False,
                                 suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Analyze Video"):

        with st.spinner("Running analysis..."):

            pipe = Pipeline(
            run_ball_tracker=True,
            run_shot_classifier=False,
            run_pose_analyzer=True,
            handedness="right",
            output_dir="output"
        )

            result = pipe.run(video_path)

        st.success("Analysis Complete")

        st.header("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
            "Technique Score",
            f"{result.technique_score:.2f}"
        )

        with col2:
            st.metric(
            "Length Zone",
            result.length_zone
        )

        with col3:
            st.metric(
            "Line Zone",
            result.line_zone
        )

        st.divider()

        st.subheader("Pose Match")

        if result.pose_flags:
            for flag in result.pose_flags:
                st.info(flag)

        st.divider()

        st.subheader("Bounce Detection")

        if result.bounce_point:
            st.write(
            f"Bounce Point: {result.bounce_point}"
        )

        if result.bounce_frame:
            st.write(
            f"Bounce Frame: {result.bounce_frame}"
        )

        st.divider()

        st.subheader("Pitch Map")

        if result.pitch_map_path and Path(result.pitch_map_path).exists():
            st.image(result.pitch_map_path)

        st.divider()

        st.subheader("Annotated Video")

        if (
        result.annotated_video_path
        and Path(result.annotated_video_path).exists()
    ):
            with open(result.annotated_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

        st.divider()

        st.subheader("AI Coach")

        try:

            coach = LLMCoach()

            tip = coach.generate(result)

            if hasattr(tip, "tip"):
                st.success(tip.tip)
            else:
                st.success(str(tip))

        except Exception as e:
            st.warning(
            f"Coach unavailable: {e}"
        )

        if result.errors:

            st.divider()

            st.subheader("Warnings")

            for err in result.errors:
                st.error(err)

