# YOLO11s Baseline Evaluation and Error Analysis

Project: NEO Streak Detection System
Model: YOLO11s
Status: Baseline established; inference settings under investigation
Evaluation approach: Validation experiments and a separate held-out test evaluation

---

## 1. Executive summary

YOLO11s was selected as the current baseline for the NEO Streak Detection System after training and comparing several YOLO model variants. On the held-out Hubble test split, the model achieved 0.915 mAP@0.5 and 0.657 mAP@0.5:0.95, with 0.933 precision and 0.825 recall under the recorded evaluation configuration.

A manual review of 30 validation images identified several recurring issues: lower-confidence detections, inaccurate bounding boxes, duplicate detections, and occasional false positives. In a specific validation example, a confidence threshold of `0.10` retained a correctly placed detection matching a ground-truth annotation, whereas `0.25` did not display that detection. A visual comparison also suggested that `0.25` produced fewer duplicate boxes in the inspected images.

The current provisional inference configuration is therefore confidence `0.10`, IoU `0.40`. This is a practical candidate for a system intended to surface possible streaks for further inspection, but it is not yet a proven optimum. A broader, consistently recorded comparison is needed to quantify the trade-off between recovering valid streaks and increasing false positives or duplicates.

## 2. Objective

The purpose of this evaluation was to establish a baseline, inspect representative prediction errors, and investigate whether inference settings could improve the usefulness of model outputs without retraining.

The evaluation focused on:

- Detection quality on the held-out test split.
- Common error patterns in validation predictions.
- The effect of confidence thresholds on visible predictions.
- Whether changing the non-maximum suppression (NMS) IoU threshold reduced duplicate boxes.

## 3. Model and training context

YOLO11s was trained using the HubbleAsteroidHunter YOLO dataset from NEOBench, with one detection class:

- Class 0: `streak`
- Image size: 640 × 640
- Batch size: 8
- Maximum epochs: 50
- Early stopping patience: 10
- Device: CUDA GPU
- Pretrained weights: enabled
- Validation split: used during training and model selection
- Test split: held out until final evaluation

The YOLO11s training run completed 50 epochs. Its best checkpoint was saved at:

`runs/detect/train-3/weights/best.pt`

The model was trained with Ultralytics YOLO. The training run used automatic optimizer selection, and mosaic augmentation was disabled for the final 10 epochs (`close_mosaic=10`).

## 4. Held-out test evaluation

The test split was evaluated after model training and validation-based selection. It contained 170 images and 170 labelled instances.

| Metric              |  Test result |
| ------------------- | -----------: |
| Precision           |        0.933 |
| Recall              |        0.825 |
| mAP@0.5             |        0.915 |
| mAP@0.5:0.95        |        0.657 |
| Preprocessing time  | 2.3 ms/image |
| Inference time      | 9.9 ms/image |
| Postprocessing time | 0.8 ms/image |

These are the recorded results from the standard Ultralytics validation command using `split=test`. The command did not explicitly set a confidence threshold of `0.10` or `0.25`; therefore, these metrics must not be presented as performance measurements for either proposed operating threshold.

### Interpretation

- Precision of 0.933 indicates strong detection precision under the evaluation matching rules.
- Recall of 0.825 indicates that some labelled streaks were not detected.
- mAP@0.5 of 0.915 indicates strong performance when a predicted box is considered a match at IoU 0.5.
- mAP@0.5:0.95 of 0.657 is lower, reflecting the stricter localisation requirements across multiple IoU thresholds.

The test results provide an independent benchmark for the current baseline. The test split should remain untouched for further threshold tuning or model selection; future experiments should use the validation split.

## 5. Manual validation error analysis

Thirty validation images were visually reviewed using predictions generated with `conf=0.05` and `iou=0.40`. The following error counts were recorded:

| Observed issue                                | Count | Percentage of 30 reviewed images\* |
| --------------------------------------------- | ----: | ---------------------------------: |
| Low-confidence but correctly placed detection |     8 |                              26.7% |
| Localisation error                            |     5 |                              16.7% |
| Duplicate detection                           |     4 |                              13.3% |
| False-positive case                           |     1 |                               3.3% |
| Missed streak                                 |     0 |                                 0% |

\*These percentages are descriptive counts from the manual sample, not formal model metrics. The categories can overlap within an image, so counts should not be added together to estimate the number of affected images. The sample is small and may not represent the full validation split.

### Error categories

Low-confidence detections. In eight observed cases, the model produced a correctly placed detection with a low confidence score. These cases motivated testing lower inference thresholds.

Localisation errors. Five cases had boxes with incorrect position or dimensions. Confidence-threshold changes do not directly fix poor box localisation; these cases may require inspection of image characteristics, annotations, and training examples.

Duplicate detections. Four cases contained multiple predicted boxes around a single streak. Non-maximum suppression can remove overlapping candidate boxes, but more aggressive suppression may also remove boxes for genuinely separate nearby streaks.

False positives. One false-positive case was observed in the sample. This is not enough to estimate the false-positive rate for the full dataset.

Missed streaks. No missed streaks were recorded in the 30-image manual sample. This does not imply that the model has perfect recall: the held-out test recall was 0.825, and a small sample may not include difficult examples.

## 6. Inference threshold experiments

The same trained YOLO11s checkpoint was evaluated on the validation split using the following explicit confidence thresholds, with IoU fixed at `0.40`.

| Confidence | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---------: | --------: | -----: | ------: | -----------: |
|       0.10 |     0.968 |  0.947 |   0.953 |        0.666 |
|       0.25 |     0.968 |  0.947 |   0.953 |        0.666 |
|       0.40 |     0.964 |  0.947 |   0.943 |        0.661 |

### Interpretation

The recorded aggregate metrics for `conf=0.10` and `conf=0.25` were identical. The `conf=0.40` run showed slightly lower precision and mAP, while recall remained the same.

These results do not demonstrate a measurable aggregate improvement from choosing `0.10` over `0.25`. Ultralytics validation metrics also depend on the validator's internal metric-calculation procedure; they should not automatically be interpreted as a direct count of all boxes visible in a saved prediction image at a given confidence setting.

The visual inspection did, however, identify a specific case where `conf=0.10` displayed a correct streak detection that was not displayed at `conf=0.25`. The streak was visually identifiable, and its ground-truth annotation covered it. This provides evidence that the lower threshold can retain a valid detection that the higher threshold filters out.

In the inspected images, `conf=0.25` also appeared to produce fewer duplicate boxes than `conf=0.10`. This was a qualitative observation rather than a quantified comparison across all validation images.

## 7. IoU / NMS investigation

With confidence fixed at `0.25`, prediction outputs were compared using IoU values of `0.30` and `0.40`.

No meaningful visual difference in duplicate detections was observed in the inspected images. Therefore, there is currently insufficient evidence that changing the IoU threshold from `0.40` to `0.30` improves the output.

The IoU threshold controls suppression of overlapping predictions during non-maximum suppression. It does not directly improve the model's learned box localisation. If duplicate boxes do not overlap enough, changing this threshold may not remove them; overly aggressive suppression can also remove legitimate detections of separate nearby streaks.

## 8. Provisional inference configuration

The current provisional setting for further development is:

```text
model: YOLO11s
weights: runs/detect/train-3/weights/best.pt
imgsz: 640
conf: 0.10
iou: 0.40
```

### Rationale

The system is intended to identify candidate streaks for further inspection. In that context, retaining a plausible, annotated streak may be preferable to suppressing it solely because its confidence is below `0.25`.

The lower threshold is supported by a concrete visual example in which a valid annotated streak was retained at `0.10` but not at `0.25`. However, the manual review also found duplicate detections, and the aggregate validation metrics were identical at `0.10` and `0.25`.

For these reasons, `0.10` is a provisional operating choice, not a demonstrated global optimum. The final setting should be chosen after a more systematic comparison of predictions at each threshold.

## 9. Limitations

- The manual error analysis covered only 30 validation images.
- Error counts were based on visual inspection and were not generated by a formal error-analysis script.
- Error categories can overlap.
- The confidence-threshold experiment produced identical aggregate metrics at `0.10` and `0.25`, despite a visible difference in at least one example.
- The visual duplicate comparison was qualitative and did not establish dataset-wide duplicate rates.
- The held-out test results were obtained using the recorded standard evaluation configuration, not a controlled benchmark of the proposed operating thresholds.
- The findings apply to the current Hubble dataset and model configuration; performance on other NEOBench datasets or real telescope data has not been established.

## 10. Recommended next steps

1. Quantify the threshold trade-off. Compare `conf=0.10` and `conf=0.25` on the same validation images, recording matched detections, missed labelled streaks, false positives, and duplicate boxes.
2. Investigate localisation errors. Compare the five observed cases against their ground-truth boxes and record whether errors concern box length, width, orientation, or position.
3. Review duplicate cases. Confirm that each apparent duplicate concerns one annotated streak rather than separate nearby streaks.
4. Keep the test split held out. Use validation data for all further threshold selection and training experiments.
5. Only then consider retraining. If localisation issues persist, investigate the relevant training examples and annotation consistency before changing training settings or selecting another model.

## 11. Conclusion

YOLO11s is a strong initial baseline for the NEO Streak Detection System, achieving a test mAP@0.5 of 0.915. Visual analysis identified lower-confidence valid detections, localisation errors, duplicate boxes, and occasional false positives. A lower confidence threshold of `0.10` recovered at least one confirmed annotated streak that was not displayed at `0.25`, while `0.25` appeared visually cleaner in terms of duplicates. Changing the NMS IoU threshold from `0.40` to `0.30` produced no meaningful visual improvement in the inspected examples.

The current working configuration is `conf=0.10` and `iou=0.40`, pending a more systematic validation comparison. The next priority is to quantify the trade-off between candidate recovery and unwanted detections, then investigate localisation errors before undertaking further training.

---

### Reproducibility notes

Baseline test evaluation command:

```bash
yolo detect val model=runs/detect/train-3/weights/best.pt data=NEOBench/NEOBench_data/datasets/HubbleAsteroidHunter_YOLO/data.yaml split=test imgsz=640 batch=8 device=0 plots=True
```

Example validation prediction command:

```bash
yolo detect predict model=runs/detect/train-3/weights/best.pt source=NEOBench/NEOBench_data/datasets/HubbleAsteroidHunter_YOLO/validation/images conf=0.10 iou=0.40 save=True
```
