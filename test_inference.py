from src.inference import NEOInference

detector = NEOInference()

detections = detector.predict(
    "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_neg_test_000752.jpg"
)

print(detections)

