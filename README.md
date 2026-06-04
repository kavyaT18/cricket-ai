# CricVis — AI-Powered Cricket Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square\&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=flat-square\&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ball_Tracking-red?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose_Estimation-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLM_Coaching-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square\&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> Upload a batting video. Get ball trajectory, bounce detection, pose-based shot analysis, pitch mapping, and AI coaching feedback — all in one dashboard.

---

## Table of Contents

* [Overview](#overview)
* [Performance Highlights](#performance-highlights)
* [Key Features](#key-features)
* [Architecture](#architecture)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Screenshots](#screenshots)
* [Future Improvements](#future-improvements)
* [Contributing](#contributing)
* [License](#license)

---

## Overview

CricVis is an end-to-end computer vision and AI analytics platform for cricket batting analysis. It processes a batting video through multiple stages including ball tracking, bounce detection, pose estimation, pitch mapping, and LLM-powered coaching feedback before presenting the results in an interactive Streamlit dashboard.

The project combines Computer Vision, Pose Estimation, Dynamic Time Warping (DTW), and Generative AI to provide actionable cricket coaching insights.

---

## Performance Highlights

* Ball tracked in **44/138 frames** on a sample batting video
* Supports **6 reference batting shot categories**
* Extracts **33 body landmarks per frame** using MediaPipe
* Classifies deliveries across **5 pitch-length zones**
* Generates AI-powered coaching feedback using Groq LLM
* End-to-end analysis completed in approximately **7–8 seconds** for a sample batting clip

---

## Key Features

### Ball Tracking & Bounce Detection

* Detects and tracks the cricket ball trajectory across video frames
* Identifies bounce frame and bounce coordinates
* Generates trajectory visualizations and annotated outputs

### Pose-Based Shot Analysis

* Uses MediaPipe Pose to extract 33 body landmarks
* Maintains reference pose embeddings for:

  * Straight Drive
  * Defense
  * Pull Shot
  * Scoop
  * Upper Cut
  * Late Cut
* Uses DTW-based pose similarity matching against reference shots
* Returns best matching shot and similarity score

### Pitch Map Analytics

* Projects bounce location onto a 2D cricket pitch model
* Classifies deliveries into:

  * Yorker
  * Full
  * Good Length
  * Short
  * Long Hop
* Generates pitch maps and bounce visualizations

### AI Coaching Assistant

* Uses Groq LLM to generate coaching feedback
* Combines:

  * Pose similarity results
  * Bounce location
  * Pitch classification
  * Shot analysis
* Produces strengths, weaknesses, and improvement suggestions

### Interactive Dashboard

* Built using Streamlit
* Upload batting videos and run complete analysis
* Displays annotated video, pitch map, analytics, and coaching feedback
* Real-time pipeline progress and results visualization

---


## Tech Stack

| Category            | Technology           |
| ------------------- | -------------------- |
| Language            | Python               |
| Dashboard           | Streamlit            |
| Computer Vision     | OpenCV               |
| Ball Detection      | YOLOv8 (Ultralytics) |
| Pose Estimation     | MediaPipe            |
| Similarity Matching | FastDTW, NumPy       |
| Visualization       | Matplotlib, Seaborn  |
| AI Coaching         | Groq LLM API         |
| Image Processing    | Pillow               |

---

## Project Structure

```text
cricvis/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── pipeline.py
│   ├── ball_tracker.py
│   ├── pose_analyzer.py
│   ├── pitch_map.py
│   └── llm_coach.py
│
├── reference_poses/
├── reference_videos/
└── output/
```

---

## Installation

### Prerequisites

* Python 3.10+
* Groq API Key

### Setup

```bash
git clone https://github.com/yourusername/cricvis.git
cd cricvis

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Usage

```bash
streamlit run app.py
```

Steps:

1. Launch the dashboard.
2. Upload a batting video.
3. Click **Analyze**.
4. Review:

   * Ball trajectory
   * Bounce detection
   * Pitch map
   * Pose similarity analysis
   * AI coaching feedback

---

## Pipeline Workflow

### Stage 1 — Ball Tracking

YOLOv8 detects the cricket ball frame-by-frame and builds a trajectory.

### Stage 2 — Bounce Detection

The trajectory is analyzed to identify the bounce frame and bounce coordinates.

### Stage 3 — Pitch Mapping

Bounce coordinates are projected onto a cricket pitch model and classified into bowling length zones.

### Stage 4 — Pose Analysis

MediaPipe extracts body landmarks which are compared against reference shot templates using Dynamic Time Warping.

### Stage 5 — AI Coaching

The analytics outputs are converted into structured prompts and passed to Groq LLM to generate coaching insights.

---

## Screenshots

### Dashboard

*Add screenshot here*

### Ball Tracking

*Add screenshot here*

### Pitch Map

*Add screenshot here*

### Pose Analysis

*Add screenshot here*

### AI Coaching Feedback

*Add screenshot here*

---

## Future Improvements

* Multi-delivery session analysis
* Aggregate pitch heatmaps across innings
* Enhanced shot classification accuracy
* Bowler release-point tracking
* Exportable PDF coaching reports
* Real-time video stream analysis
* Larger reference shot library
* Fine-tuned cricket-specific detection models

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

<p align="center">
Built with OpenCV, MediaPipe, YOLOv8, DTW, Streamlit, and Groq.
</p>
