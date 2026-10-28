# NEO Streak Detection System

An AI-powered computer vision project using the NEOBench dataset to detect candidate streaks in astronomical images. The project aims to develop an interactive application for visualising detections, evaluating model performance, and exploring future extensions towards automated near-Earth object analysis.

## Project Overview

Near-Earth object (NEO) detection is an important application of astronomical observation and computer vision. This project explores how object detection techniques can identify streak-like features in astronomical images for further investigation.

The system is being developed around a structured machine learning pipeline, from dataset analysis and preparation through model training and evaluation to an interactive application for submitting images and viewing detection results.

A detected streak is a candidate for further investigation, not confirmation that the object is an asteroid or near-Earth object.

## Project Objectives

- Investigate and assess the quality of astronomical image datasets.
- Develop a reliable pipeline for loading images and YOLO annotations.
- Prepare and augment data for object detection.
- Research, train, and evaluate suitable detection models.
- Build reusable inference functionality for processing images.
- Develop an interactive application for uploading images and viewing predictions.
- Store and retrieve detection results through a database.
- Test, document, and demonstrate the completed system.

## Proposed Application Architecture

The initial design separates the user interface, backend, inference pipeline, and data storage.

```text
User
  |
  v
User Interface
  |
  v
Backend
  |
  v
Inference Pipeline
  |
  +--> Preprocessing
  |
  +--> Trained Detection Model
  |
  +--> Postprocessing
  |
  v
Detection Results
  |
  +-----------> User Interface
  |
  +-----------> Database
                   |
                   v
             Detection History
```

This is a proposed architecture and may evolve during development. The backend will coordinate image processing and inference, while the database will support persistent storage and retrieval of detection results.

## Development Progress

The project is organised into phases covering research, dataset preparation, model development, application engineering, and final testing.

### Phase 1 — Research and Project Design

- [x] Research the NEO dataset.
- [x] Inspect sample images and annotations.
- [x] Define the machine learning problem.
- [x] Set up the development environment.
- [x] Design the initial application architecture.
- [x] Establish the initial data pipeline.

### Phase 2 — Dataset Analysis and Preparation

- [x] Dataset analysis.
- [x] Data quality assessment.
- [x] Dataset preparation.

### Phase 3 — Model Research and Baseline

- [x] Research candidate models.
- [x] Define the baseline strategy.
- [x] Implement the baseline.
- [x] Evaluate baseline performance.

### Phase 4 — Model Training and Evaluation

- [x] Training and fine-tuning.
- [x] First training run.
- [x] Training analysis.
- [x] Model experiments.
- [ ] Final test set evaluation. (in progress)
- [ ] Error analysis. (in progress)
- [ ] Final model selection. (in progress)

### Phase 5 — Inference Pipeline

- [ ] Build reusable inference.
- [ ] Define the detection result format.

### Phase 6 — Backend and Data Storage

- [ ] Establish the backend foundation.
- [ ] Implement image upload.
- [ ] Build the prediction API.
- [ ] Design the database.
- [ ] Store detection results.
- [ ] Retrieve stored results.

### Phase 7 — Application Development

- [ ] Design the application layout.
- [ ] Build the upload interface.
- [ ] Display detection results.
- [ ] Implement detection history.
- [ ] Implement detection details.
- [ ] Add error handling.
- [ ] Optimise application performance.

### Phase 8 — Testing and Delivery

- [ ] Test the machine learning pipeline.
- [ ] Clean and organise the repository.
- [ ] Complete project documentation and README.
- [ ] Prepare the final demonstration.

## Technology Stack

| Technology       | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| Python           | Primary development language                   |
| OpenCV           | Image loading and processing                   |
| Matplotlib       | Image visualisation and bounding-box plotting  |
| Ultralytics YOLO | Object detection framework under consideration |
| PyTorch          | Deep learning framework                        |
| NEOBench         | Astronomical streak-detection dataset          |

### Planned Application Technologies

The initial application architecture proposes the following technologies, which have not yet been implemented:

- **React and TypeScript** — frontend interface.
- **FastAPI** — backend and prediction API.
- **Database** — technology to be determined during development.

The final technology stack will be refined as the project progresses.

## Dataset

The project uses NEOBench, a dataset collection containing five source datasets for astronomical streak detection.

The project is investigating YOLO-formatted annotations, where bounding boxes describe labelled objects using a class ID and normalised centre coordinates, width, and height.

Dataset analysis and quality assessment will establish the distribution of images and annotations across splits, identify potential data issues, and determine the preparation needed before training.

Detailed dataset statistics and preparation decisions will be documented as the analysis progresses.

## Project Structure

The repository is being organised to separate application code, dataset analysis, documentation, and supporting resources.

The directory structure will be documented here once the main components and their responsibilities have stabilised.

## Setup and Usage

The project uses a Python virtual environment and a set of project dependencies.

Installation instructions, dataset access requirements, and commands for running the analysis tools and application will be added as the workflow becomes established.

## Evaluation Strategy

Model selection will be based on measured performance rather than visual examples alone. Evaluation will consider appropriate object detection metrics, including precision, recall, and mean average precision (mAP).

The final test set will be reserved for evaluation after model development and selection. Error analysis will help identify common failure cases, including missed streaks and false detections.

## Limitations and Scope

- Detecting a streak does not establish that the streak is an asteroid or NEO.
- Model performance depends on dataset quality, annotation consistency, and the diversity of the training data.
- Detection results will require careful interpretation and may need further astronomical verification.
- The final model, application framework, and database technology have not yet been selected.

## Development Notes

Significant technical decisions, dataset limitations, implementation challenges, and architectural changes will be recorded as the project progresses.

## Demonstration

A project demonstration, including example images, detection visualisations, and evaluation results, will be added once the end-to-end application is working.
