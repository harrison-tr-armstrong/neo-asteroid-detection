1. Objective: Detect candidate streaks in uploaded astronomical images
2. Interface: React + TypeScript.
3. Backend: FastAPI REST API.
4. AI component: A trained YOLO model using your NEOBench dataset.
5. Data flow: Upload → validate → infer → return results → display detections.
6. Future scope: Batch processing, storage, automated ingestion and agent-assisted workflows.
7. Design principle: Keep model inference independent of the interface.
