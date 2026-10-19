# Foreign Object Debris (FOD) Detection Report

Model: YOLO11s fine-tuned on a small subset of FOD-A. Trained on a Kaggle GPU (the local RTX 2050 crashed repeatedly with `CUDA error: unknown error`).
Files: `results/` (annotated images, result video, curves). Code: `code/kaggle_fod.ipynb` and `code/kaggle_fod.py`.

**How to read the numbers.** *Exact* = read from the training/eval files. *≈* = read off a plot by eye (the notebook's printed test metrics line was not saved). *Not available* = not recorded. It is listed in "Open items" at the end rather than invented.

---

## 1. Dataset (Task 1)

**FOD-A v2.1, Pascal VOC release** (github.com/FOD-UNOmaha/FOD-data).
- 33,793 images, **31 object classes** (tools, fasteners, luggage parts, wire, rock, wood, soda can …), 300×300 px, one bounding box per object. Light-level (bright/dim/dark) and wet/dry weather tags exist as separate files. I did not use them in this run.
- **Quality / imbalance observed in EDA** (`results/eda.png`):
  - *Imbalance:* in the used subset the most common class (Bolt) has ≈400 instances and the rarest (Screw, FuelCap) ≈80, a 5:1 spread.
  - *Tiny objects:* the box-area histogram is dominated by very small boxes. On the full set ≈27 % of boxes cover <1 % of the image.
  - *Mostly one object per frame* (≈98 % of images), always on a plain concrete/asphalt surface, usually centred.
  - *Near-duplicates:* frames are video frames, so consecutive images are almost identical.
  - Cleaning: dropped unreadable images, exact (MD5) duplicates, boxes <2 px or outside the frame, images with no valid box. On the data I checked locally the script found **no** such defects (0 removed), so the labels are structurally clean. Label *correctness* was not audited by hand.
- **Subset:** to fit the time budget I used **12 % of each class's frame blocks** (`FRAC=0.12`, **4,420 images**, all 31 classes).
- **Split (train/val/test):** done **per class in blocks of 20 consecutive frame ids**, whole blocks assigned to one split (seed 42), with at least one val and one test block per class. Result: **train 2,860 / val 780 / test 780 images** (65 % / 18 % / 18 %), with 2,940 / 780 / 780 object instances. Why: (a) adjacent frames are near-identical, so a random per-image split leaks test images into training and inflates mAP. The dataset's own `test.txt` is a random frame sample and has this problem. (b) frames are stored grouped by class, so a naive split would leave whole classes unseen in training. **Test set = 780 images.**
- **Caveats:** the block size assumes ids follow video order. The same scene/background can still appear in several blocks. Because the test set holds *different video segments* of each class, it is a fair (somewhat hard) generalisation test, which partly explains the class confusions below.

## 2. Training (Task 2)

| Setting | Value | Why |
|---|---|---|
| Model | YOLO11s, COCO-pretrained | Transfer learning suits a small dataset, small enough for a Jetson-class onboard computer |
| Image size | 512 | Source images are 300×300, 512 (~1.7× upsample) keeps tiny objects visible at lower cost than 640 |
| Epochs / batch | 15 (early stop 6) / 32 | Time budget on a free GPU. Validation mAP was still creeping up at epoch 15, so the model is **under-trained** |
| Optimiser | AdamW, lr 1e-3, cosine decay | Stable fine-tuning |
| Augmentation | HSV jitter (h 0.015, s 0.6, v 0.5), rotation ±15°, flips (up-down and left-right), scale 0.6, translate 0.1, mosaic (off for last 3 epochs), mixup 0.1, small perspective | HSV → lighting / time-of-day / weather, flips + rotation → a UAV looking down has no canonical "up", scale → different flight altitudes / zoom, mosaic/mixup → more object contexts and regularisation on a tiny dataset |
| Caching | RAM | Remove CPU image-decoding bottleneck |

Training curves: `results/results.png`. Final epoch, **validation** set (exact, from `results.csv`): precision **0.907**, recall **0.783**, mAP@50 **0.854**, mAP@50-95 **0.680**.

## 3. Detection results on the held-out test set (Task 3)

Inference at conf = 0.25, IoU-match 0.5. Colours: **green = true positive (with confidence)**, **red = false positive**, **blue = missed ground truth (false negative)**. Result video: `results/result.mp4` (test frames, boxes + labels + confidence).

Image-level tally over the 780 test images (a miss takes priority over a false alarm): **602 clean (all objects found, no extras), 71 with a false alarm only, 107 with a miss.** A "miss" here includes an object that was boxed but given the **wrong class**.

| # | Image | What it shows |
|---|---|---|
| 1 | `1_TP_pliers.jpg` | Correct detection, Pliers 0.89 |
| 2 | `2_FP_wire_duplicate_boxes.jpg` | Wire found, but extra overlapping/partial boxes → false positives |
| 3 | `3_FP_shoe_unlabelled.jpg` | A shoe at the frame edge is detected, it is a real unlabelled object |
| 4 | `4_FN_rock_missed.jpg` | Rock, no box at all (true miss) |
| 5 | `5_FN_screw_tiny_wrong_class.jpg` | Very small Screw, predicted as ClampPart → counts as FP + FN |
| 6 | `6_FN_paintchip_class_confusion.jpg` | PaintChip localised correctly but labelled PlasticPart (0.73) |

## 4. Error analysis (Task 4)

Which classes fail (image-level, from the test gallery, each class is 20-60 test images):
- **Never or almost never correct:** ClampPart 20/20 missed, Rock 20/20, Screw 20/20, PaintChip 18/20, BoltNutSet 17/27.
- **Most false alarms:** Wire 19 and Bolt 19 (of 40 / 60 images), Label 11, MetalPart 9, Hose 5.
- **About 20 classes were perfect** at this threshold (e.g. Pliers, Wrench, Hammer, Cutter, Battery, Nail, Wood).

**Why the misses happened (from the images):**
1. **Wrong class, not missing object (dominant).** In images 5 and 6 the box is in the right place but the label is wrong (PaintChip→PlasticPart, Screw→ClampPart). The confusion matrix (`confusion_matrix_normalized.png`) shows the same: heavy off-diagonal mass between visually similar classes (e.g. Cutter↔AdjustableWrench, Hose↔Wood, PaintChip↔MetalSheet, Rock↔SodaCan). Likely causes: **very few training examples per class (~100) from a few video segments**, and classes that look alike in 300×300 crops (dark flat fragments, metal pieces). Test blocks are *different video segments*, so the model has not seen those viewpoints. (This is my interpretation of the images and matrix, I did not run an ablation to prove it.)
2. **Small size.** The Screw is a few pixels wide. At 300 px source resolution there is little detail to classify.
3. **Truncation at the frame edge** (the ClampPart example is cut by the border).
4. **Rare / hard classes** (Rock, Screw, FuelCap have the fewest instances). Rock is a *true* miss: low contrast, brown-grey on grey concrete, texture similar to the background.

**Why false positives fired (from the images):**
1. **Duplicate / partial boxes on one long object** (Wire, Label): the same object gets several overlapping boxes of different classes, an NMS / class-confusion problem, not a background problem.
2. **Real objects missing from the labels** (the shoe, image 3): the detector is arguably right and the dataset is incomplete. Counted as FP, but operationally a legitimate alert.
3. Wrong-class boxes on real objects (counted as FP + FN).

**Honest limit of this analysis:** the task suggests shadows, runway markings, reflections and surface texture as FP sources. **I did not observe those**, because FOD-A images are close-up shots of one object on clean, uniform concrete, no markings, no structure shadows, no wet glare. This is a major dataset limitation: the low false-alarm rate here **will not transfer** to a real runway. I would expect markings, rubber streaks, cracks, joint sealant and lights to cause many more false alarms in deployment, and cannot quantify that from this data.

**Miss vs false alarm, which is worse?** A **miss is worse**: an undetected bolt or tyre fragment can destroy an engine or burst a tyre, while a false alarm costs an inspection stop. But false alarms are not free: too many make operators ignore the system. So the policy is *recall-first, then suppress false alarms downstream* (not by pushing the model threshold very high).

**Threshold tuning:** choose the operating point on the **validation** set (not test), prefer high recall, and tune per class (lower for dangerous/hard classes). Then reduce false alarms with N-of-M frame confirmation, tracking, and a human check of the alert crop. A **class-agnostic "debris" detector** is the better safety design, runway safety cares that *something* is there, not whether it is a Screw or a ClampPart. Collapsing the 31 classes into one would turn most of today's "misses" (wrong class) into correct detections. A second stage can classify afterwards. (Not tested here.)

**Improvements with more time/data (priority order):**
1. **Class-agnostic detection first, classification second**, plus a hard-negative set (empty runway with markings, shadows, cracks, rubber marks, wet surface, night/fog) so false alarms are measured and trained against.
2. **More data and training:** full FOD-A (about 8× more than the subset), more epochs, per-class oversampling / loss weighting for the rare classes (Rock, Screw, ClampPart, PaintChip), and more viewpoints/altitudes to close the unseen-segment gap.
3. **Small-object handling:** higher-resolution input or tiled (SAHI-style) inference. Class-agnostic NMS to remove duplicate-box false positives.

## 5. Metrics (Task 5)

**Test set** (780 images). The source of each number is marked.

| Metric | Value | Source |
|---|---|---|
| mAP@50 | **0.844** | exact, printed in the PR-curve legend |
| mAP@50-95 | not available for test, **validation = 0.680** | val value exact |
| Precision | not recorded at the chosen conf, **val = 0.907**, test **≈0.82 at conf 0.25, ≈0.89 at conf 0.485** | val exact, test ≈ read off `BoxP_curve.png` |
| Recall | not recorded at the chosen conf, **val = 0.783**, test **≈0.83 at conf 0.25, ≈0.79 at conf 0.485** | val exact, test ≈ read off `BoxR_curve.png` |
| Best F1 | **0.78 at conf 0.485** | exact, F1-curve legend |
| Maximum recall (conf → 0) | **≈0.89** | R-curve legend |

Figures (all in `results/`): PR curve `BoxPR_curve.png`, F1 `BoxF1_curve.png`, precision/recall-confidence `BoxP_curve.png`, `BoxR_curve.png`, confusion matrix `confusion_matrix_normalized.png`.

**Reading the curves.** The mean PR curve is high and flat (precision ≈0.9 up to recall ≈0.7) and drops only near recall ≈0.9, but the thin grey per-class lines show a handful of classes with very poor curves (the same ClampPart / Rock / Screw / PaintChip group). The recall curve tops out near **0.89 even at confidence 0**, so ≈11 % of test objects are never found-and-classified correctly at any threshold. **lowering the threshold cannot fix those, only more data or a class-agnostic detector can.**

**Chosen confidence threshold: 0.25** for the safety use case (a deliberate departure from the F1-optimal 0.485):
- F1 is nearly flat between ≈0.2 and ≈0.6, so lowering to 0.25 costs very little F1.
- recall rises from ≈0.79 to ≈0.83 (+~4 points) at the price of precision ≈0.89 → ≈0.82.
- below ≈0.15 recall barely improves but precision falls steeply, so the false-alarm burden grows fast.
- the extra false alarms are handled downstream (N-of-M frames, tracking, operator check), as described in §4.
This is a recall-biased operating point chosen from the curves. For a certification-grade claim it should be set on validation data against a target recall (e.g. ≥0.95 for hazardous classes), which this model cannot yet reach.

## Extra points: altitude and weather/lighting
**Not done.** I did not train or test across altitudes or weather/lighting. Only scale and HSV augmentations were used as proxies, which is *not* evidence of robustness. FOD-A provides light-level and wet/dry tags. The natural next step is to report metrics per tag (bright/dim/dark, wet/dry) and per box-size bucket (a stand-in for altitude).

## 6. UAV deployment architecture

1. **Sensing:** gimbal-stabilised RGB camera (add thermal/IR for night and fog), flying a planned low-altitude sweep along the runway. GPS/RTK + IMU for geotagging. Altitude is chosen so the smallest hazardous object spans a minimum number of pixels (this dataset's tiny-object problem means the ground sampling distance must be specified up front).
2. **Onboard compute:** Jetson Orin-class module. Export the model to **TensorRT FP16/INT8**. Tiled/sliced inference for small objects. Target ≥10-15 FPS.
3. **Pipeline:** capture → preprocess → class-agnostic FOD detector → tracker (e.g. ByteTrack) → temporal filter (alert only after N of M frames) → classifier on the crop → geo-projection of the box to runway coordinates using camera pose + altitude.
4. **Comms / ground station:** alert with crop, class, confidence and map position over a datalink (LTE/5G/telemetry) to the airport operations dashboard. Operator confirms and dispatches ground crew. All evidence logged.
5. **Safety & ops:** recall-biased thresholds with human in the loop. Geofencing and ATC coordination. Battery / return-to-home failsafes. Fall back to manual inspection when conditions are outside the model's validated range (night, fog, heavy rain).
6. **MLOps loop:** confirmed alerts and false alarms feed back as new labelled data and hard negatives. Periodic retraining with versioned models and per-condition (weather/light/altitude) monitoring for drift.

## Open items (not hidden)
- Test mAP@50-95, test precision/recall at the chosen conf and exact TP/FP/FN instance counts were printed in the Kaggle notebook but not saved. Pasting the printed lines (cells 2, 4, 5) would replace the "≈" values above with exact ones.
- No altitude or weather experiment. No runway-like negatives. The model is under-trained and uses a 12 % subset.
- Labels were not manually audited.
