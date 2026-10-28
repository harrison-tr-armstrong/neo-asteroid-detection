Pretrained model: yolo11s.pt
Dataset: Hubble Asteroid Hunter
Classes: 1 - streak
Image size: 640 x 640
Batch size: 8
Max epochs: 50
Early stopping: Patience 10
Initial learning rate: 0.01
Final learning rate factor: 0.01
Optimizer: auto
Weight decay: 0.0005
Warmup: 3 epochs
Mosaic augmentation: 1.0
Horizontal flip: 0.5
HSV augmentation: 0.015 / 0.7 / 0.4
Translation: 0.1
Scale: 0.5
Rotation: 0
Mixup: 0
Validation: Enabled
Hardware: RTX 3050, 8gb VRAM


My configuration uses several forms of augmentation. Which create variations of training images to help the model learn patterns that aren't tied to one image appearance.

Enabled in my config:
- Mosaic - extra detail: close_mosaic was set to 10 which allowed the augmentation to not run and be switched off during the final 10 training epochs, this helps the model finish on images closer to their normal appearance.
- Horizontal flipping
- Hue, saturation and brightness adjustments
- Translation and scaling
- RandAugment, as the configured auto augmentation policy

disabled in config:
- Rotation
- Shear and perspective
- Vertical flipping
- Mixup and CutMix
- copy & paste augmentation


### Final Test Evaluation — YOLO11s

Checkpoint: `runs/detect/train-3/weights/best.pt`

Dataset: Hubble Asteroid Hunter YOLO
Split: Test
Images evaluated: 170
Labelled instances: 170

| Metric | Test result |
|---|---:|
| Precision | 0.933 |
| Recall | 0.825 |
| mAP50 | 0.915 |
| mAP50–95 | 0.657 |
| Preprocessing time | 2.3 ms/image |
| Inference time | 9.9 ms/image |
| Postprocessing time | 0.8 ms/image |

Findings: YOLO11s achieved strong detection performance on the held-out test set. All four principal detection metrics were lower than the validation results, with the largest reduction observed in recall. This indicates that the model missed a greater proportion of labelled streaks on the test split. The results support YOLO11s as a strong baseline candidate, while further investigation of missed detections and generalisation remains worthwhile.

Evaluation configuration: Image size 640, batch size 8, GPU device 0. Metrics were obtained using the standard Ultralytics validation configuration. The inference IoU setting was not explicitly overridden for this evaluation.

Checkpoint status: The trained checkpoint was evaluated without further training or tuning against the test set.
