import pytest
from src.inference import NEOInference
import src.inference as inference_module
from unittest.mock import Mock
import numpy as np
from PIL import Image


# --------------------------------------------------
# FIXTURES
# --------------------------------------------------

@pytest.fixture
def detector():
    # Creates a real NEOInference instance using the default model.
    # Used when we want to test actual model inference.
    return NEOInference()


@pytest.fixture
def mocked_detector(monkeypatch):
    # Creates a mock YOLO model so tests don't need to load
    # an actual model checkpoint or perform real inference.
    fake_model = Mock()

    # Replaces YOLO inside the inference module with a function
    # that returns our fake model instead.
    monkeypatch.setattr(
        inference_module,
        "YOLO",
        lambda model_path: fake_model
    )

    return NEOInference()


@pytest.fixture
def mocked_yolo(monkeypatch):
    # Creates a mock for the YOLO model constructor.
    # Allows us to verify that the model is loaded correctly.
    mock_yolo = Mock()

    monkeypatch.setattr(
        inference_module,
        "YOLO",
        mock_yolo
    )

    return mock_yolo


@pytest.fixture
def mocked_prediction(mocked_detector):
    # Creates a simulated YOLO detection with a bounding box,
    # class ID and confidence score.
    fake_box = Mock()
    fake_box.xyxy = [np.array([100, 150, 300, 350])]
    fake_box.cls = [np.array(0)]
    fake_box.conf = [np.array(0.95)]

    # Simulates the result returned by YOLO inference.
    fake_result = Mock()
    fake_result.boxes = [fake_box]
    fake_result.names = {0: "streak"}

    # Makes the mocked model return our simulated detection.
    mocked_detector.model.predict.return_value = [fake_result]

    return mocked_detector


# --------------------------------------------------
# BATCH INFERENCE TESTS
# --------------------------------------------------

def test_empty_batch(mocked_detector):

    # An empty list should not be accepted as a batch.
    # The method should raise a ValueError.
    with pytest.raises(ValueError):
        mocked_detector.predict_batch([])


def test_batch_success(detector):

    # Uses a real image and the actual model to test
    # whether batch inference successfully processes an image.
    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    results = detector.predict_batch([image_path])

    # Checks that one result is returned with the expected structure.
    assert len(results) == 1
    assert results[0]["status"] == "success"
    assert results[0]["image"] == image_path
    assert "detections" in results[0]
    assert "annotated_image" in results[0]


def test_batch_continues_after_invalid_image(detector):

    # Creates a batch containing valid and invalid image paths.
    valid_image = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"
    invalid_image = "does_not_exist.jpg"

    results = detector.predict_batch([
        valid_image,
        invalid_image,
        valid_image
    ])

    # Verifies that a failed image does not prevent the remaining
    # images in the batch from being processed.
    assert len(results) == 3
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "error"
    assert results[2]["status"] == "success"


def test_invalid_file_extension(tmp_path, mocked_detector):

    # Creates a temporary text file to simulate an unsupported image.
    invalid_image = tmp_path / "example.txt"
    invalid_image.write_text("This is not an image.")

    results = mocked_detector.predict_batch([invalid_image])

    # Checks that the invalid file is reported as an error
    # rather than being processed as an image.
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "Unsupported image format." in results[0]["error"]


# --------------------------------------------------
# INPUT VALIDATION TESTS
# --------------------------------------------------

@pytest.mark.parametrize("conf", [-0.5, 1.5])
def test_invalid_confidence(conf, mocked_detector):

    # Tests confidence values outside the valid range of 0 to 1.
    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(ValueError):
        mocked_detector.predict(image_path, conf=conf)

    # Invalid parameters should be rejected before model inference.
    mocked_detector.model.predict.assert_not_called()


@pytest.mark.parametrize("iou", [-0.1, 1.5])
def test_invalid_iou(iou, mocked_detector):

    # Tests IoU thresholds outside the valid range of 0 to 1.
    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(ValueError):
        mocked_detector.predict(image_path, iou=iou)

    mocked_detector.model.predict.assert_not_called()


def test_invalid_imgsz_type(mocked_detector):

    # Image size must be a supported numeric value,
    # not a string such as "large".
    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(TypeError):
        mocked_detector.predict(image_path, imgsz="large")

    mocked_detector.model.predict.assert_not_called()


def test_invalid_imgsz_value(mocked_detector):

    # Image dimensions must be positive.
    # A negative image size should raise a ValueError.
    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(ValueError):
        mocked_detector.predict(image_path, imgsz=-640)

    mocked_detector.model.predict.assert_not_called()


def test_invalid_imgsz_boolean(mocked_detector):

    # Python treats bool as a subclass of int.
    # This test ensures True is not accidentally accepted as an image size.
    image_path = "NEOBench/NEOBench_data/datasets/UAEMMN_YOLO/test/images/uaemmn_pos_test_000004.jpg"

    with pytest.raises(TypeError):
        mocked_detector.predict(image_path, imgsz=True)

    mocked_detector.model.predict.assert_not_called()


# --------------------------------------------------
# PREDICTION RESULT TESTS
# --------------------------------------------------

def test_successful_prediction(mocked_prediction, tmp_path):

    # Creates a temporary image for the prediction test.
    image_path = tmp_path / "test_image.jpg"
    Image.new("RGB", (640, 640)).save(image_path)

    # Runs prediction without saving an annotated image.
    results = mocked_prediction.predict(
        image_path,
        save_annotated=False
    )

    # Confirms that exactly one detection was returned.
    assert len(results["detections"]) == 1

    detection = results["detections"][0]

    # Checks that each detection contains the expected fields.
    assert set(detection.keys()) == {"class", "confidence", "bounding_box"}

    # Verifies the data types and values of the detection.
    assert isinstance(detection["class"], str)
    assert isinstance(detection["confidence"], float)
    assert detection["class"] == "streak"
    assert detection["confidence"] == 0.95
    assert isinstance(detection["bounding_box"], dict)

    # Ensures the bounding box contains all four coordinates.
    assert set(detection["bounding_box"].keys()) == {
        "x1",
        "y1",
        "x2",
        "y2"
    }

    # Checks that the coordinates match the simulated YOLO output.
    assert detection["bounding_box"] == {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 350
    }

    # Confirms that bounding box coordinates are numeric.
    assert all(
        isinstance(coord, (int, float))
        for coord in detection["bounding_box"].values()
    )

    # Saving was disabled, so no annotated image should be returned.
    assert results["annotated_image"] is None

    # Confirms that the model was called exactly once.
    mocked_prediction.model.predict.assert_called_once()


def test_multiple_detections(mocked_detector, tmp_path):

    # Creates a temporary image containing simulated detections.
    image_path = tmp_path / "multiple_streaks.jpg"
    Image.new("RGB", (640, 640)).save(image_path)

    # Creates the first simulated streak detection.
    first_box = Mock()
    first_box.xyxy = [np.array([100, 150, 300, 350])]
    first_box.cls = [np.array(0)]
    first_box.conf = [np.array(0.95)]

    # Creates a second detection with a different position and confidence.
    second_box = Mock()
    second_box.xyxy = [np.array([400, 100, 550, 250])]
    second_box.cls = [np.array(0)]
    second_box.conf = [np.array(0.85)]

    # Combines both detections into one simulated YOLO result.
    fake_result = Mock()
    fake_result.boxes = [first_box, second_box]
    fake_result.names = {0: "streak"}

    mocked_detector.model.predict.return_value = [fake_result]

    results = mocked_detector.predict(
        image_path,
        save_annotated=False
    )

    detections = results["detections"]

    # Verifies that both detections are returned independently.
    assert len(detections) == 2

    # Checks that each detection retains its correct confidence.
    assert detections[0]["confidence"] == pytest.approx(0.95)
    assert detections[1]["confidence"] == pytest.approx(0.85)

    # Checks that the bounding boxes belong to the correct detections.
    assert detections[0]["bounding_box"]["x1"] == 100
    assert detections[1]["bounding_box"]["x1"] == 400


def test_no_detections(mocked_detector, tmp_path):

    # Creates a temporary image with no simulated detections.
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

    # An image without detections should return an empty list,
    # not an error.
    assert results["detections"] == []
    assert results["annotated_image"] is None


def test_unknown_class_id(mocked_detector, tmp_path):

    # Creates a detection with class ID 1, which is not present
    # in the model's class-name mapping.
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

    # Accessing an unknown class ID should raise a KeyError
    # because the class cannot be mapped to a name.
    with pytest.raises(KeyError):
        mocked_detector.predict(
            image_path,
            save_annotated=False
        )


def test_batch_multiple_predictions(mocked_detector, tmp_path):

    # Creates two temporary images to test multiple successful predictions.
    image_paths = []

    for i in range(2):
        image_path = tmp_path / f"image_{i}.jpg"
        Image.new("RGB", (640, 640)).save(image_path)
        image_paths.append(image_path)

    # Creates a separate simulated detection for each image.
    first_box = Mock()
    first_box.xyxy = [np.array([100, 150, 300, 350])]
    first_box.cls = [np.array(0)]
    first_box.conf = [np.array(0.95)]

    second_box = Mock()
    second_box.xyxy = [np.array([400, 100, 550, 250])]
    second_box.cls = [np.array(0)]
    second_box.conf = [np.array(0.85)]

    # Creates individual YOLO results for each image.
    first_result = Mock()
    first_result.boxes = [first_box]
    first_result.names = {0: "streak"}
    first_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

    second_result = Mock()
    second_result.boxes = [second_box]
    second_result.names = {0: "streak"}
    second_result.plot.return_value = np.zeros((640, 640, 3), dtype=np.uint8)

    # side_effect provides a different result each time predict() is called.
    mocked_detector.model.predict.side_effect = [
        [first_result],
        [second_result]
    ]

    results = mocked_detector.predict_batch(image_paths)

    # Both images should be processed successfully.
    assert len(results) == 2
    assert all(result["status"] == "success" for result in results)

    # Each image should contain exactly one detection.
    assert len(results[0]["detections"]) == 1
    assert len(results[1]["detections"]) == 1

    # Checks that the detections have not been mixed between images.
    assert results[0]["detections"][0]["confidence"] == pytest.approx(0.95)
    assert results[1]["detections"][0]["confidence"] == pytest.approx(0.85)

    # Confirms that inference was performed once for each image.
    assert mocked_detector.model.predict.call_count == 2


# --------------------------------------------------
# ANNOTATED IMAGE TESTS
# --------------------------------------------------

def test_save_prediction(mocked_detector, tmp_path):

    # Creates a blank NumPy image to simulate YOLO's annotated output.
    annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)

    image_path = tmp_path / "example.png"
    output_dir = tmp_path / "predictions"

    # Saves the annotated image using the inference class.
    output_path = mocked_detector.save_prediction(
        annotated_image,
        image_path,
        output_dir
    )

    # Verifies that the output file exists at the expected location.
    assert output_path.exists()
    assert output_path == output_dir / "example_prediction.jpg"

    # Opens the saved image and checks its dimensions.
    saved_image = Image.open(output_path)

    assert saved_image.size == (100, 100)


def test_save_prediction_colour_conversion(mocked_detector, tmp_path):

    # Creates an image containing a pure red pixel in OpenCV's BGR format.
    annotated_image = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)

    image_path = tmp_path / "example.png"

    output_path = mocked_detector.save_prediction(
        annotated_image,
        image_path,
        tmp_path
    )

    # Reopens the saved image as an RGB NumPy array.
    saved_image = np.array(Image.open(output_path))

    # Confirms that the BGR-to-RGB conversion preserved the red colour.
    assert saved_image[0, 0][0] > 200
    assert saved_image[0, 0][1] < 50
    assert saved_image[0, 0][2] < 50


def test_annotate(mocked_detector):

    fake_result = Mock()
    fake_annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Simulates the annotated image returned by YOLO's plot() method.
    fake_result.plot.return_value = fake_annotated_image

    annotated_image = mocked_detector.annotate(fake_result)

    # Ensures annotate() delegates to YOLO's plot() method exactly once.
    fake_result.plot.assert_called_once()

    # Confirms that the returned image is the same object produced by plot().
    assert annotated_image is fake_annotated_image


def test_predict_saves_annotated_image(mocked_prediction, tmp_path):

    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    output_dir = tmp_path / "predictions"
    output_path = output_dir / "example_prediction.jpg"

    annotated_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Replaces annotation and saving methods with mocks.
    # This isolates the test to the prediction workflow.
    mocked_prediction.annotate = Mock(return_value=annotated_image)
    mocked_prediction.save_prediction = Mock(return_value=output_path)

    result = mocked_prediction.predict(
        image_path,
        save_annotated=True,
        output_dir=output_dir
    )

    # Confirms that the YOLO result was passed to annotate().
    mocked_prediction.annotate.assert_called_once_with(
        mocked_prediction.model.predict.return_value[0]
    )

    # Confirms that the annotated image was passed to the saving method.
    mocked_prediction.save_prediction.assert_called_once_with(
        annotated_image,
        image_path,
        output_dir
    )

    # The returned result should contain the saved image's path.
    assert result["annotated_image"] == str(output_path)


def test_predict_without_saving(mocked_prediction, tmp_path):

    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    # Replaces annotation and saving methods with mocks so we can
    # verify that neither is called when saving is disabled.
    mocked_prediction.annotate = Mock()
    mocked_prediction.save_prediction = Mock()

    result = mocked_prediction.predict(
        image_path,
        save_annotated=False
    )

    # Confirms that annotation and file saving are skipped.
    mocked_prediction.annotate.assert_not_called()
    mocked_prediction.save_prediction.assert_not_called()

    # Prediction should still return its detections normally.
    assert len(result["detections"]) == 1
    assert result["annotated_image"] is None


# --------------------------------------------------
# ERROR HANDLING TESTS
# --------------------------------------------------

def test_model_prediction_error(mocked_detector, tmp_path):

    image_path = tmp_path / "example.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    # Simulates an error occurring inside the YOLO model.
    mocked_detector.model.predict.side_effect = RuntimeError(
        "Model inference failed"
    )

    # Confirms that the error is propagated to the caller.
    with pytest.raises(RuntimeError, match="Model inference failed"):
        mocked_detector.predict(image_path)

    mocked_detector.model.predict.assert_called_once()


def test_batch_stops_on_model_error(mocked_detector, tmp_path):

    # Creates three images to test how batch inference handles
    # an unexpected model error during processing.
    image_paths = []

    for i in range(3):
        image_path = tmp_path / f"image_{i}.jpg"
        Image.new("RGB", (100, 100)).save(image_path)
        image_paths.append(image_path)

    # Creates a successful result for the first image.
    fake_result = Mock()
    fake_result.boxes = []
    fake_result.names = {0: "streak"}
    fake_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # The first inference succeeds, but the second raises an error.
    mocked_detector.model.predict.side_effect = [
        [fake_result],
        RuntimeError("Model inference failed")
    ]

    # Confirms that the unexpected model error propagates out of
    # predict_batch() rather than being silently ignored.
    with pytest.raises(RuntimeError, match="Model inference failed"):
        mocked_detector.predict_batch(image_paths)

    # Only two calls should occur: the successful first image
    # and the failing second image. The third image is never processed.
    assert mocked_detector.model.predict.call_count == 2


# --------------------------------------------------
# MODEL INITIALISATION TESTS
# --------------------------------------------------

def test_missing_model_checkpoint(tmp_path, monkeypatch):

    # Creates a path pointing to a checkpoint that does not exist.
    missing_model = tmp_path / "missing_model.pt"

    # Initialising the inference class with a missing checkpoint
    # should raise a FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        NEOInference(model_path=missing_model)


def test_model_checkpoint_loaded(mocked_yolo):

    # Initialises the inference class using the mocked YOLO constructor.
    detector = NEOInference()

    # Verifies that YOLO was called with the configured model checkpoint.
    mocked_yolo.assert_called_once_with(
        str(detector.model_path)
    )
