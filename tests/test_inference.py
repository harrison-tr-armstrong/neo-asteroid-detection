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
def mocked_yolo(monkeypatch):
    mock_yolo = Mock()

    monkeypatch.setattr(
        inference_module,
        "YOLO",
        mock_yolo
    )

    return mock_yolo


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

    assert set(detection.keys()) == {"class", "confidence", "bounding_box"}

    assert isinstance(detection["class"], str)
    assert isinstance(detection["confidence"], float)
    assert detection["class"] == "streak"
    assert detection["confidence"] == 0.95
    assert isinstance(detection["bounding_box"], dict)

    assert set(detection["bounding_box"].keys()) == {
      "x1",
      "y1",
      "x2",
      "y2"
   }

    assert detection["bounding_box"] == {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 350
    }

    assert all(
      isinstance(coord, (int, float))
      for coord in detection["bounding_box"].values()
   )

    assert results["annotated_image"] is None
    mocked_prediction.model.predict.assert_called_once()



def test_multiple_detections(mocked_detector, tmp_path):
    image_path = tmp_path / "multiple_streaks.jpg"
    Image.new("RGB", (640, 640)).save(image_path)

    first_box = Mock()
    first_box.xyxy = [np.array([100, 150, 300, 350])]
    first_box.cls = [np.array(0)]
    first_box.conf = [np.array(0.95)]

    second_box = Mock()
    second_box.xyxy = [np.array([400, 100, 550, 250])]
    second_box.cls = [np.array(0)]
    second_box.conf = [np.array(0.85)]

    fake_result = Mock()
    fake_result.boxes = [first_box, second_box]
    fake_result.names = {0: "streak"}

    mocked_detector.model.predict.return_value = [fake_result]

    results = mocked_detector.predict(
        image_path,
        save_annotated=False
    )

    detections = results["detections"]

    assert len(detections) == 2

    assert detections[0]["confidence"] == pytest.approx(0.95)
    assert detections[1]["confidence"] == pytest.approx(0.85)

    assert detections[0]["bounding_box"]["x1"] == 100
    assert detections[1]["bounding_box"]["x1"] == 400



def test_no_detections(mocked_detector, tmp_path):
    image_path = tmp_path / "empty_image.jpg"
    Image.new("RGB", (640, 640)).save(image_path)

    fake_result = Mock()
    fake_result.boxes = []
    fake_result.names = {0: "streak"}

    mocked_detector.model.predict.return_value = [fake_result]

    results = mocked_detector.predict(
        image_path,
        save_annotated=False
    )

    assert results["detections"] == []
    assert results["annotated_image"] is None



def test_unknown_class_id(mocked_detector, tmp_path):
    image_path = tmp_path / "unknown_class.jpg"
    Image.new("RGB", (640, 640)).save(image_path)

    fake_box = Mock()
    fake_box.xyxy = [np.array([100, 150, 300, 350])]
    fake_box.cls = [np.array(1)]
    fake_box.conf = [np.array(0.95)]

    fake_result = Mock()
    fake_result.boxes = [fake_box]
    fake_result.names = {0: "streak"}

    mocked_detector.model.predict.return_value = [fake_result]

    with pytest.raises(KeyError):
        mocked_detector.predict(
            image_path,
            save_annotated=False
        )



def test_batch_multiple_predictions(mocked_detector, tmp_path):
    image_paths = []

    for i in range(2):
        image_path = tmp_path / f"image_{i}.jpg"
        Image.new("RGB", (640, 640)).save(image_path)
        image_paths.append(image_path)

    first_box = Mock()
    first_box.xyxy = [np.array([100, 150, 300, 350])]
    first_box.cls = [np.array(0)]
    first_box.conf = [np.array(0.95)]

    second_box = Mock()
    second_box.xyxy = [np.array([400, 100, 550, 250])]
    second_box.cls = [np.array(0)]
    second_box.conf = [np.array(0.85)]

    first_result = Mock()
    first_result.boxes = [first_box]
    first_result.names = {0: "streak"}
    first_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

    second_result = Mock()
    second_result.boxes = [second_box]
    second_result.names = {0: "streak"}
    second_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

    mocked_detector.model.predict.side_effect = [
        [first_result],
        [second_result]
    ]

    results = mocked_detector.predict_batch(image_paths)

    assert len(results) == 2
    assert all(result["status"] == "success" for result in results)

    assert len(results[0]["detections"]) == 1
    assert len(results[1]["detections"]) == 1

    assert results[0]["detections"][0]["confidence"] == pytest.approx(0.95)
    assert results[1]["detections"][0]["confidence"] == pytest.approx(0.85)

    assert mocked_detector.model.predict.call_count == 2



def test_save_prediction(mocked_detector, tmp_path):

    annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)

    image_path = tmp_path / "example.png"
    output_dir = tmp_path / "predictions"

    output_path = mocked_detector.save_prediction(
        annotated_image,
        image_path,
        output_dir
    )

    assert output_path.exists()
    assert output_path == output_dir / "example_prediction.jpg"

    saved_image = Image.open(output_path)

    assert saved_image.size == (100, 100)


def test_save_prediction_colour_conversion(mocked_detector, tmp_path):

    annotated_image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)

    image_path = tmp_path / "example.png"

    output_path = mocked_detector.save_prediction(
        annotated_image,
        image_path,
        tmp_path
    )

    saved_image = np.array(Image.open(output_path))

    assert saved_image[0, 0][0] > 200
    assert saved_image[0, 0][1] < 50
    assert saved_image[0, 0][2] < 50


def test_annotate(mocked_detector):
    fake_result = Mock()
    fake_annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)

    fake_result.plot.return_value = fake_annotated_image

    annotated_image = mocked_detector.annotate(fake_result)

    fake_result.plot.assert_called_once()
    assert annotated_image is fake_annotated_image


def test_predict_saves_annotated_image(mocked_prediction, tmp_path):
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    output_dir = tmp_path / "predictions"
    output_path = output_dir / "example_prediction.jpg"

    annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)

    mocked_prediction.annotate = Mock(return_value=annotated_image)
    mocked_prediction.save_prediction = Mock(return_value=output_path)

    result = mocked_prediction.predict(
        image_path,
        save_annotated=True,
        output_dir=output_dir
    )

    mocked_prediction.annotate.assert_called_once_with(
        mocked_prediction.model.predict.return_value[0]
    )

    mocked_prediction.save_prediction.assert_called_once_with(
        annotated_image,
        image_path,
        output_dir
    )

    assert result["annotated_image"] == str(output_path)


def test_predict_without_saving(mocked_prediction, tmp_path):
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    mocked_prediction.annotate = Mock()
    mocked_prediction.save_prediction = Mock()

    result = mocked_prediction.predict(
        image_path,
        save_annotated=False
    )

    mocked_prediction.annotate.assert_not_called()
    mocked_prediction.save_prediction.assert_not_called()

    assert len(result["detections"]) == 1
    assert result["annotated_image"] is None


def test_model_prediction_error(mocked_detector, tmp_path):
    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    mocked_detector.model.predict.side_effect = RuntimeError(
        "Model inference failed"
    )

    with pytest.raises(RuntimeError, match="Model inference failed"):
        mocked_detector.predict(image_path)

    mocked_detector.model.predict.assert_called_once()


def test_batch_stops_on_model_error(mocked_detector, tmp_path):
    image_paths = []

    for i in range(3):
        image_path = tmp_path / f"image_{i}.jpg"
        Image.new("RGB", (100, 100)).save(image_path)
        image_paths.append(image_path)

    fake_result = Mock()
    fake_result.boxes = []
    fake_result.names = {0: "streak"}
    fake_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    mocked_detector.model.predict.side_effect = [
        [fake_result],
        RuntimeError("Model inference failed")
    ]

    with pytest.raises(RuntimeError, match="Model inference failed"):
        mocked_detector.predict_batch(image_paths)

    assert mocked_detector.model.predict.call_count == 2


def test_missing_model_checkpoint(tmp_path, monkeypatch):
    missing_model = tmp_path / "missing_model.pt"

    with pytest.raises(FileNotFoundError):
        NEOInference(model_path=missing_model)


def test_model_checkpoint_loaded(mocked_yolo):
    detector = NEOInference()

    mocked_yolo.assert_called_once_with(
        str(detector.model_path)
    )
