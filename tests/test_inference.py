import pytest
from src.inference import NEOInference
import src.inference as inference_module
from unittest.mock import Mock
import numpy as np
from PIL import Image

@pytest.fixture
def detector():
    return NEOInference()

@pytest.fixture
def mocked_detector(monkeypatch):
    fake_model = Mock()

    monkeypatch.setattr(
        inference_module,
        "YOLO",
        lambda model_path: fake_model
    )

    return NEOInference()


@pytest.fixture
def mocked_prediction(mocked_detector):

    fake_box = Mock()
    fake_box.xyxy = [np.array([100, 150, 300, 350])]
    fake_box.cls = [np.array(0)]
    fake_box.conf = [np.array(0.95)]

    fake_result = Mock()
    fake_result.boxes = [fake_box]
    fake_result.names = {0: "streak"}

    mocked_detector.model.predict.return_value = [fake_result]

    return mocked_detector


def test_empty_batch(mocked_detector):

   with pytest.raises(ValueError):
      mocked_detector.predict_batch([])


def test_batch_success(detector):

    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    results = detector.predict_batch([image_path])

    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["image"] == image_path
    assert "detections" in results[0]
    assert "annotated_image" in results[0]


def test_batch_continues_after_invalid_image(detector):

    valid_image = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"
    invalid_image = "does_not_exist.jpg"

    results = detector.predict_batch([
        valid_image,
        invalid_image,
        valid_image
    ])

    assert len(results) == 3
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "error"
    assert results[2]["status"] == "success"


def test_invalid_file_extension(tmp_path, mocked_detector):

    invalid_image = tmp_path / "example.txt"
    invalid_image.write_text("This is not an image.")

    results = mocked_detector.predict_batch([invalid_image])

    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "Unsupported image format." in results[0]["error"]


@pytest.mark.parametrize("conf", [-0.5, 1.5])
def test_invalid_confidence(conf, mocked_detector):

    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(ValueError):
          mocked_detector.predict(image_path, conf=conf)

    mocked_detector.model.predict.assert_not_called()


@pytest.mark.parametrize("iou", [-0.1, 1.5])
def test_invalid_iou(iou, mocked_detector):

    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(ValueError):
        mocked_detector.predict(image_path, iou=iou)

    mocked_detector.model.predict.assert_not_called()


def test_invalid_imgsz_type(mocked_detector):

    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(TypeError):
        mocked_detector.predict(image_path, imgsz="large")

    mocked_detector.model.predict.assert_not_called()


def test_invalid_imgsz_value(mocked_detector):

    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(ValueError):
        mocked_detector.predict(image_path, imgsz=-640)

    mocked_detector.model.predict.assert_not_called()


def test_invalid_imgsz_boolean(mocked_detector):

    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(TypeError):
        mocked_detector.predict(image_path, imgsz=True)

    mocked_detector.model.predict.assert_not_called()


def test_successful_prediction(mocked_prediction, tmp_path):

    image_path = tmp_path / "test_image.jpg"

    Image.new("RGB", (640, 640)).save(image_path)

    results = mocked_prediction.predict(
        image_path,
        save_annotated=False
    )

    assert len(results["detections"]) == 1

    detection = results["detections"][0]

    assert detection["class"] == "streak"
    assert detection["confidence"] == 0.95
    assert detection["bounding_box"] == {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 350
    }

    assert results["annotated_image"] is None
    mocked_prediction.model.predict.assert_called_once()
