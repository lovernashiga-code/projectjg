# Results: lfw

62 people, 2063 aligned faces. Test set: 70/30 stratified split of the images (seed 42).

## Accuracy (%) by test variant

| Variant | M0 Eigenfaces+SVM | M1 VGG-Face full | M2 VGG-Face full+partial | M3 FaceNet full | M4 FaceNet full+partial |
|---|---:|---:|---:|---:|---:|
| full (ใบหน้าเต็ม) | 56.2 | 99.7 | 99.8 | 99.8 | 99.7 |
| upper_half (ครึ่งบน) | 5.2 | 90.8 | 98.4 | 94.7 | 97.7 |
| lower_half (ครึ่งล่าง) | 4.7 | 36.7 | 85.8 | 60.9 | 88.5 |
| eyes_only (เฉพาะดวงตา) | 2.1 | 44.4 | 91.6 | 69.1 | 92.6 |
| nose_mouth (เฉพาะจมูก+ปาก) | 1.3 | 36.7 | 89.3 | 55.1 | 91.4 |
| mask (สวมหน้ากากอนามัย) | 3.9 | 85.3 | 97.7 | 93.5 | 97.6 |
| sunglasses (สวมแว่นกันแดด) | 7.1 | 77.9 | 95.5 | 89.0 | 97.9 |

## M2: all metrics

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Top-5 | Coverage@τ | Accuracy@τ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.998 | 0.999 | 0.998 | 0.998 | 1.000 | 0.998 | 0.992 | 1.000 |
| upper_half | 0.984 | 0.986 | 0.983 | 0.984 | 1.000 | 0.998 | 0.964 | 0.998 |
| lower_half | 0.858 | 0.863 | 0.845 | 0.848 | 0.994 | 0.964 | 0.761 | 0.943 |
| eyes_only | 0.916 | 0.929 | 0.909 | 0.912 | 0.998 | 0.982 | 0.819 | 0.988 |
| nose_mouth | 0.893 | 0.901 | 0.880 | 0.884 | 0.997 | 0.979 | 0.793 | 0.961 |
| mask | 0.977 | 0.981 | 0.978 | 0.979 | 1.000 | 0.998 | 0.950 | 1.000 |
| sunglasses | 0.955 | 0.957 | 0.947 | 0.949 | 0.999 | 0.994 | 0.927 | 0.984 |

## Targets (slide 5)

| | Target | Measured | |
|---|---|---|---|
| T1 Accuracy on full faces, M2 | ≥ 0.95 | 0.998 | PASS |
| T2 Accuracy with mask / sunglasses, M2 | ≥ 0.85 | mask 0.977, sunglasses 0.955 | PASS |
| T3 Acc(M2) - Acc(M1) on the variant where M1 is worst | +0.1 | lower_half +0.491, nose_mouth +0.527 | PASS |

## Data augmentation (slide 6)

Accuracy (%) without and with 2 augmented copies of every training image (rotation ±15°, random crop, flip, brightness/contrast, noise; test images never augmented).

| Variant | M1 | M1+aug | M2 | M2+aug | M3 | M3+aug | M4 | M4+aug |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 99.7 | 99.7 | 99.8 | 99.8 | 99.8 | 99.8 | 99.7 | 99.8 |
| upper_half | 90.8 | 91.0 | 98.4 | 98.5 | 94.7 | 94.0 | 97.7 | 97.6 |
| lower_half | 36.7 | 35.7 | 85.8 | 84.3 | 60.9 | 63.0 | 88.5 | 87.6 |
| eyes_only | 44.4 | 47.5 | 91.6 | 91.1 | 69.1 | 71.6 | 92.6 | 92.2 |
| nose_mouth | 36.7 | 37.2 | 89.3 | 90.1 | 55.1 | 57.2 | 91.4 | 90.0 |
| mask | 85.3 | 85.9 | 97.7 | 98.1 | 93.5 | 94.0 | 97.6 | 97.3 |
| sunglasses | 77.9 | 75.8 | 95.5 | 95.0 | 89.0 | 89.2 | 97.9 | 97.3 |

## Strangers (M2, open-set test)

500 photos of people who are not in the training set, tested like the 619 known test faces. *Accepted* = given a name instead of "Unknown".

| Variant | τ = 0.5: known, right name | known, wrong name | strangers accepted | τ for ≤ 5 % strangers accepted | known, right name at that τ |
|---|---:|---:|---:|---:|---:|
| full | 99.2 % | 0.0 % | 7.4 % | 0.55 | 99.0 % |
| upper_half | 96.3 % | 0.2 % | 14.0 % | 0.70 | 91.9 % |
| lower_half | 71.7 % | 4.4 % | 32.6 % | 0.85 | 40.7 % |
| eyes_only | 80.9 % | 1.0 % | 13.6 % | 0.70 | 67.2 % |
| nose_mouth | 76.3 % | 3.1 % | 20.6 % | 0.75 | 56.5 % |
| mask | 95.0 % | 0.0 % | 13.4 % | 0.70 | 88.7 % |
| sunglasses | 91.3 % | 1.5 % | 27.6 % | 0.85 | 70.1 % |

## Figures

- [accuracy_by_variant.png](figures/accuracy_by_variant.png)
- [augmentation.png](figures/augmentation.png)
- [augmentation_effect.png](figures/augmentation_effect.png)
- [class_distribution.png](figures/class_distribution.png)
- [confusion_full_M1_lower_half.png](figures/confusion_full_M1_lower_half.png)
- [confusion_full_M2_lower_half.png](figures/confusion_full_M2_lower_half.png)
- [confusion_top12_M1_lower_half.png](figures/confusion_top12_M1_lower_half.png)
- [confusion_top12_M2_lower_half.png](figures/confusion_top12_M2_lower_half.png)
- [examples_correct_wrong.png](figures/examples_correct_wrong.png)
- [open_set_M2.png](figures/open_set_M2.png)
- [roc_M2.png](figures/roc_M2.png)
- [roc_mask_methods.png](figures/roc_mask_methods.png)
- [samples_per_class.png](figures/samples_per_class.png)
- [threshold_M2.png](figures/threshold_M2.png)
- [variants.png](figures/variants.png)
