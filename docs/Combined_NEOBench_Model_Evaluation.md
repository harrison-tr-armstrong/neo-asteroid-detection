# Combined NEOBench Model Evaluation

Project: NEO Streak Detection System
Model: YOLO11s
Dataset: Combined NEOBench
Evaluation type: Object detection
Status: Evaluation completed; final model selection pending

---

## 1. Overview

Following the initial testing of different YOLO model versions and sizes, I decided to move forward with a combined training approach using all five datasets included in NEOBench.

The original baseline model was trained using only the Hubble Asteroid Hunter dataset. Although it produced good results, I wanted to see whether training with the wider collection of datasets could improve the model's ability to detect streaks across different astronomical images.

The combined model was trained using YOLO11s, with all five datasets included in the training, validation and test configuration.

The main objective was to produce a model that could reliably identify streaks across the different datasets, rather than performing particularly well on just one dataset.

The combined model produced strong overall results, with a test mAP50 of 0.980 and mAP50-95 of 0.908. Individual dataset evaluations also showed that performance was generally strong, although there were some differences between datasets.

This document covers the training configuration, dataset composition, evaluation results, comparison against the original baseline and visual observations made during prediction testing.

## 2. Dataset composition

The combined dataset consists of five individual YOLO-compatible datasets, all using the same class mapping:

| Class ID | Class  |
| -------- | ------ |
| 0        | streak |

The datasets were provided as part of the larger NEOBench collection and were prepared in a compatible YOLO format.

### Dataset breakdown

| Dataset                | Training images |
| ---------------------- | --------------: |
| Hubble Asteroid Hunter |           1,361 |
| Roboflow               |             929 |
| StreakMind             |           1,484 |
| StreaksYoloDataset     |           1,722 |
| UAEMMN                 |          11,834 |
| Total                  |          17,330 |

The combined training configuration uses all five datasets rather than creating a separate physical dataset containing duplicated image files.

UAEMMN makes up a substantial proportion of the training data, accounting for approximately 68.3% of the training images. This is worth considering when looking at the final results, as the model has been exposed to considerably more training images from UAEMMN than from the other datasets.

### Dataset configuration

A separate configuration file was created:

`combined_neobench.yaml`

This file contains the paths to the training, validation and test image directories for all five datasets, alongside the shared class mapping.

This allowed Ultralytics to load the five datasets through one configuration file and train a single model using the combined data.

## 3. Training configuration

The combined model was trained using the pretrained YOLO11s weights as its starting point.

The intention was to take advantage of the existing pretrained model and fine-tune it for the specific task of detecting astronomical streaks.

### Training settings

| Setting                 | Value                  |
| ----------------------- | ---------------------- |
| Model                   | YOLO11s                |
| Starting weights        | yolo11s.pt             |
| Training configuration  | combined_neobench.yaml |
| Maximum epochs          | 50                     |
| Early stopping patience | 10                     |
| Batch size              | 8                      |
| Image size              | 640 × 640              |
| Device                  | CUDA GPU (device 0)    |
| Random seed             | 0                      |
| Deterministic training  | Enabled                |
| Task                    | Object detection       |
| Number of classes       | 1                      |
| Class mapping           | 0: streak              |

The model was trained for a maximum of 50 epochs, with early stopping enabled to prevent unnecessary training once improvements had stopped.

The final model used for evaluation was the best checkpoint saved during training:

`runs/detect/runs/detect/yolo11s_neobench_combined/weights/best.pt`

The `best.pt` checkpoint was used for all subsequent evaluation and prediction runs.

## 4. Combined validation results

After training, the model was evaluated against the combined validation split.

This provides an indication of how well the model performs on data that was not used to update its weights during training.

### Validation metrics

| Metric    | Result |
| --------- | -----: |
| Images    |  2,341 |
| Instances |  1,414 |
| Precision |  0.971 |
| Recall    |  0.946 |
| mAP50     |  0.974 |
| mAP50-95  |  0.878 |

### Understanding the results

The validation results were strong across all four evaluation metrics.

Precision of 0.971 indicates that approximately 97.1% of the model's matched detection predictions were correct under the evaluation settings.

Recall of 0.946 indicates that the model successfully detected approximately 94.6% of the labelled streak instances.

The mAP50 score of 0.974 shows that the model performed very well when detections were considered correct at an IoU threshold of 0.50.

The mAP50-95 score of 0.878 is also strong. This metric evaluates detection performance across a range of IoU thresholds from 0.50 to 0.95, making it a stricter measure of how accurately the model places its bounding boxes.

The difference between mAP50 and mAP50-95 suggests that performance becomes more challenging when tighter bounding-box accuracy is required. However, the overall result remains strong.

### Validation speed

| Processing stage | Time per image |
| ---------------- | -------------: |
| Preprocessing    |         0.8 ms |
| Inference        |         4.4 ms |
| Postprocessing   |         0.7 ms |

The measured inference time was approximately 4.4 ms per image during this validation run.

These timings are specific to the evaluation environment and should not be treated as guaranteed application response times.

## 5. Combined test results

The combined test split was used to evaluate the trained model on data that was kept separate from training and validation.

This is particularly important because the test results provide a more independent indication of how well the model performs on the held-out data.

### Test metrics

| Metric    | Result |
| --------- | -----: |
| Images    |  2,117 |
| Instances |  1,254 |
| Precision |  0.991 |
| Recall    |  0.949 |
| mAP50     |  0.980 |
| mAP50-95  |  0.908 |

### Interpretation

The combined test results were slightly higher than the validation results for precision, mAP50 and mAP50-95.

The model achieved:

- 99.1% precision
- 94.9% recall
- 98.0% mAP50
- 90.8% mAP50-95

The precision score indicates that false-positive detections were relatively uncommon across the combined test split.

Recall was slightly lower than precision, meaning the model missed some labelled streaks rather than frequently producing incorrect detections.

The mAP50-95 result of 0.908 is particularly encouraging because it shows that the model maintained strong performance even when more accurate bounding-box placement was required.

It is worth noting that these results represent the combined test split. Individual datasets can have different levels of difficulty, so the overall metrics do not tell the complete story on their own.

### Test speed

| Processing stage | Time per image |
| ---------------- | -------------: |
| Preprocessing    |         1.2 ms |
| Inference        |         7.6 ms |
| Postprocessing   |         0.6 ms |

The measured inference time was approximately 7.6 ms per image during this test evaluation.

## 6. Individual dataset test results

To get a better understanding of how the combined model performed across the different sources, I evaluated the model separately against each dataset's test split.

This was useful because a strong combined result could potentially hide weaker performance on individual datasets.

### Results

| Dataset            | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------------------ | -----: | --------: | --------: | -----: | ----: | -------: |
| Hubble             |    170 |       170 |     0.968 |  0.878 | 0.941 |    0.630 |
| Roboflow           |     40 |        76 |     0.958 |  0.909 | 0.939 |    0.797 |
| StreakMind         |     94 |        43 |     1.000 |  0.805 | 0.917 |    0.504 |
| StreaksYoloDataset |    333 |       225 |     0.979 |  0.916 | 0.949 |    0.889 |
| UAEMMN             |  1,480 |       740 |     1.000 |  0.996 | 0.995 |    0.995 |

### Hubble

Hubble produced a precision score of 0.968 and recall of 0.878.

The mAP50 score of 0.941 indicates that the model was generally successful at identifying streaks. However, its mAP50-95 score of 0.630 was noticeably lower than the overall combined result.

This suggests that although the model could usually identify the presence of a streak, bounding-box placement was less consistent when evaluated using stricter IoU thresholds.

During visual inspection, I also noticed some examples where the model detected the correct streak but did not place the bounding box as accurately as I would have liked.

Hubble is therefore one of the main areas where further investigation could be useful.

### Roboflow

Roboflow produced strong results, with precision of 0.958 and recall of 0.909.

Its mAP50-95 score of 0.797 was higher than Hubble and StreakMind, suggesting more consistent bounding-box placement.

The test split contained 40 images and 76 instances, so the results should be considered in the context of its relatively small sample size.

Overall, the model performed well on this dataset.

### StreakMind

StreakMind produced a precision score of 1.000 and recall of 0.805.

This means the model's matched predictions were highly accurate, but it missed a larger proportion of labelled streaks than on some of the other datasets.

Its mAP50 score was 0.917, while mAP50-95 was 0.504.

The difference between these two scores suggests that accurate localisation was more difficult on this dataset.

StreakMind also had the smallest number of test instances, with just 43. This makes its metrics more sensitive to the outcome of individual detections.

Although the results are still useful, I would avoid drawing overly strong conclusions from this dataset alone.

### StreaksYoloDataset

StreaksYoloDataset produced consistently strong results.

Precision reached 0.979, recall was 0.916 and mAP50-95 reached 0.889.

This was the second-highest mAP50-95 score across the individual datasets, behind UAEMMN.

The results suggest that the model was not only identifying streaks successfully but was also placing its bounding boxes accurately.

### UAEMMN

UAEMMN produced the strongest individual results.

The model achieved:

- Precision: 1.000
- Recall: 0.996
- mAP50: 0.995
- mAP50-95: 0.995

The evaluation included 1,480 images and 740 labelled instances.

These results indicate that the model performed exceptionally well on this particular test split, with very few missed instances and no false-positive detections reported by the evaluation at its operating settings.

UAEMMN also makes up a large proportion of the combined test split, representing approximately 69.9% of its images.

This means its strong performance has a considerable influence on the combined test metrics. It is therefore important to look at the individual dataset results rather than relying solely on the combined score.

## 7. Comparison against the original Hubble-only baseline

Before creating the combined model, I trained a YOLO11s model using only the Hubble Asteroid Hunter dataset.

This provided an initial baseline that could be compared against the combined model using the same Hubble test split.

Both models were evaluated against the same 170 Hubble test images and 170 instances.

### Direct comparison

| Metric    | Hubble-only baseline | Combined model | Difference |
| --------- | -------------------: | -------------: | ---------: |
| Precision |                0.933 |          0.968 |     +0.035 |
| Recall    |                0.825 |          0.878 |     +0.053 |
| mAP50     |                0.915 |          0.941 |     +0.026 |
| mAP50-95  |                0.657 |          0.630 |     -0.027 |

### What changed?

The combined model improved precision, recall and mAP50 when evaluated against the original Hubble test split.

The largest improvement was in recall, increasing from 0.825 to 0.878. This suggests that the combined model was able to identify more of the labelled streaks.

Precision also improved, indicating that this increase in detections did not come at the cost of producing more incorrect predictions overall.

However, mAP50-95 decreased slightly, from 0.657 to 0.630.

This suggests that while the combined model became better at identifying streaks, its bounding-box accuracy across stricter IoU thresholds was not quite as strong on Hubble.

I consider this a reasonable trade-off at this stage, particularly because the combined model was trained to work across five datasets rather than just Hubble.

The comparison also shows why it is important to consider multiple metrics instead of selecting a model based on a single score.

## 8. Visual prediction review

Alongside the numerical evaluation, I generated prediction images using the combined model and visually inspected examples from Hubble and StreakMind.

The purpose was to check whether the metrics reflected what the model was actually detecting.

The prediction runs used the following settings:

| Setting              |    Value |
| -------------------- | -------: |
| Confidence threshold |     0.10 |
| IoU threshold        |     0.40 |
| Image size           |      640 |
| Device               | CUDA GPU |

The confidence and IoU settings were selected for visual inspection based on earlier experiments. They should not be interpreted as a fully optimised configuration for every dataset.

### Observations

Missed streaks

Only a small number of missed streaks were noticed during the initial inspection.

The examples I found were generally very faint streaks, which were difficult to distinguish from the background.

This suggests that low-visibility streaks remain one of the more challenging cases for the model.

False positives

I identified one example where the model detected the actual streak correctly but also placed an additional box around an unrelated part of the image.

This is worth recording because it shows that the model can occasionally identify an unrelated image feature as a possible streak, even when it has correctly detected the target elsewhere in the same image.

Bounding-box localisation

Most of the predictions I inspected had good bounding-box placement.

However, I noticed some examples in Hubble where the model identified the correct streak but the bounding box was not perfectly positioned.

This is consistent with Hubble's lower mAP50-95 score compared with some of the other datasets.

Duplicate detections

I did not notice any significant pattern of duplicate detections during the initial inspection.

The additional false-positive box mentioned above was separate from the actual streak detection rather than an overlapping duplicate.

Successful detections

Most of the predictions inspected were good.

One particularly encouraging observation was that the model identified some faint streaks that I initially found difficult to notice myself.

This suggests that the model is picking up some of the smaller or less obvious visual features associated with streaks.

### Visual review limitations

This was an initial qualitative inspection rather than a formal statistical error analysis.

The images were selected for inspection, and the number of images reviewed was not recorded as a controlled sample.

The observations are therefore useful for identifying potential weaknesses, but they should not be treated as measured false-positive rates, missed-detection rates or localisation error percentages.

A more structured review can be carried out later if further model improvement is required.

## 9. Main findings

Looking across the validation results, test results and visual inspection, there are several points worth highlighting.

### Strengths

- The combined model achieved strong overall test performance, with mAP50 of 0.980 and mAP50-95 of 0.908.
- Precision and recall were both high across the combined test split.
- Performance was particularly strong on UAEMMN and StreaksYoloDataset.
- The combined model improved precision, recall and mAP50 compared with the original Hubble-only baseline on the same Hubble test images.
- Visual inspection showed generally accurate streak detection with very few obvious errors in the images reviewed.
- The model runs inference using the available GPU without requiring unusually large hardware resources.

### Areas for improvement

- Very faint streaks can still be missed.
- Bounding-box localisation is less consistent on Hubble and StreakMind.
- StreakMind produced a noticeably lower mAP50-95 score than the other datasets.
- Occasional false-positive detections are still possible.
- UAEMMN contributes a large proportion of the combined dataset, so its strong performance has a significant influence on the overall results.

These findings do not necessarily mean the model needs another training run. They provide a starting point for deciding whether further changes would offer a meaningful improvement.

## 10. Limitations and considerations

Although the results are strong, there are some limitations to keep in mind.

Dataset coverage

The model has been trained and evaluated using the five datasets included in NEOBench. Strong performance on these datasets does not automatically guarantee the same performance on completely new astronomical images.

Dataset balance

The training data is not evenly distributed. UAEMMN accounts for approximately 68.3% of the training images, meaning it contributes substantially more examples than the other datasets.

This should be considered when interpreting the individual and combined results.

Bounding-box accuracy

The difference between mAP50 and mAP50-95 shows that detection performance depends partly on how strictly the bounding boxes are evaluated.

Identifying a streak and accurately outlining it are related but separate challenges.

Visual inspection

The visual review was relatively small and informal. It helped identify some obvious issues but does not provide a complete measurement of model errors.

Real-world testing

The model has not yet been fully integrated into an application or tested as a complete user-facing system.

Inference speed measured during Ultralytics evaluation may differ from the response time of a finished application, where image loading, preprocessing, API requests and other operations also contribute.

## 11. Current model assessment

Based on the results collected so far, the combined YOLO11s model is my leading candidate for the NEO Streak Detection System.

It provides strong detection performance across the five datasets, has improved several metrics compared with the original Hubble-only baseline and has shown good results during visual inspection.

The slightly weaker localisation results on Hubble and StreakMind are worth investigating, but I do not currently have enough evidence to justify another training run.

I would rather keep the current model as a strong candidate and focus on making it usable through a consistent inference pipeline.

The next stage will be to build an inference script that loads the trained weights, accepts an image and returns the detected streaks with their bounding-box coordinates and confidence scores.

This will move the project beyond model experimentation and towards a working application.

## 12. Conclusion

The combined NEOBench training approach produced a clear improvement in the overall capability of the original Hubble-only model when considering performance across the wider collection of datasets.

The final combined test evaluation achieved 0.991 precision, 0.949 recall, 0.980 mAP50 and 0.908 mAP50-95.

Individual evaluations showed that performance was not identical across all datasets, with particularly strong results on UAEMMN and StreaksYoloDataset, while Hubble and StreakMind presented more challenges with accurate bounding-box placement.

Visual inspection also supported the numerical results, with most of the reviewed predictions being accurate and only a small number of obvious errors identified.

At this point, I consider the combined YOLO11s model a strong candidate for the next stage of development.

Rather than immediately introducing more complex models or additional training, the priority is to build a reliable inference pipeline, test it with new images and eventually integrate it into the wider application.

---

Evaluation checkpoint: Combined NEOBench YOLO11s model
Checkpoint used: `best.pt`
Next development stage: Reusable inference pipeline (`inference.py`)
