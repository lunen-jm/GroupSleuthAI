# Group Dynamics Analysis Project

This project analyzes group dynamics in images using computer vision techniques. It combines face detection, emotion recognition, gaze estimation, and spatial proximity analysis to infer relationships and interactions between people in group photos.

## Features

- **Face Detection**: Detects and localizes faces in images
- **Emotion Recognition**: Identifies emotions of detected faces
- **Gaze Estimation**: Determines where each person is looking
- **Spatial Proximity Analysis**: Analyzes physical distances between people
- **Group Clustering**: Segments people into social groups
- **Relationship Analysis**: Infers potential relationships based on emotion, gaze, and proximity

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download required datasets (see notebooks for details)
4. Run the notebooks in order

## Project Structure

- `notebooks/`: Jupyter notebooks for each component
- `src/`: Source code for reusable functions
- `data/`: Datasets and test images
- `models/`: Trained models
- `results/`: Analysis outputs and visualizations