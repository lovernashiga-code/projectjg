# Results: lfw

62 people, 2063 aligned faces, 70/30 stratified split (seed 42).

## Accuracy (%) by test variant

| Variant | M0 Eigenfaces+SVM | M1 VGG-Face full | M2 VGG-Face full+partial | M3 FaceNet full | M4 FaceNet full+partial |
|---|---:|---:|---:|---:|---:|
| full (ใบหน้าเต็ม) | 59.5 | 99.7 | 99.8 | 99.8 | 99.7 |
| upper_half (ครึ่งบน) | 7.0 | 90.8 | 98.4 | 94.7 | 97.7 |
| lower_half (ครึ่งล่าง) | 4.0 | 36.7 | 85.1 | 60.9 | 88.7 |
| eyes_only (เฉพาะดวงตา) | 2.4 | 44.6 | 91.1 | 69.1 | 93.2 |
| nose_mouth (เฉพาะจมูก+ปาก) | 1.1 | 36.7 | 89.3 | 55.4 | 91.6 |
| mask (สวมหน้ากากอนามัย) | 5.7 | 85.3 | 97.2 | 93.5 | 97.7 |
| sunglasses (สวมแว่นกันแดด) | 6.6 | 77.9 | 94.8 | 89.0 | 97.7 |

## M2: all metrics

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Top-5 | Coverage@τ | Accuracy@τ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.998 | 0.998 | 0.998 | 0.998 | 1.000 | 0.998 | – | – |
| upper_half | 0.984 | 0.986 | 0.983 | 0.984 | 1.000 | 0.998 | – | – |
| lower_half | 0.851 | 0.855 | 0.839 | 0.841 | 0.994 | 0.968 | – | – |
| eyes_only | 0.911 | 0.925 | 0.904 | 0.907 | 0.998 | 0.982 | – | – |
| nose_mouth | 0.893 | 0.901 | 0.883 | 0.885 | 0.997 | 0.977 | – | – |
| mask | 0.973 | 0.977 | 0.972 | 0.973 | 1.000 | 0.998 | – | – |
| sunglasses | 0.948 | 0.950 | 0.940 | 0.942 | 0.999 | 0.994 | – | – |

## Targets (slide 5)

| | Target | Measured | |
|---|---|---|---|
| T1 Accuracy on full faces, M2 | ≥ 0.95 | 0.998 | PASS |
| T2 Accuracy with mask / sunglasses, M2 | ≥ 0.85 | mask 0.973, sunglasses 0.948 | PASS |
| T3 Acc(M2) - Acc(M1) on the variant where M1 is worst | +0.1 | lower_half +0.485, nose_mouth +0.527 | PASS |

## Figures

- [accuracy_by_variant.png](figures/accuracy_by_variant.png)
- [class_distribution.png](figures/class_distribution.png)
- [confusion_full_M1_lower_half.png](figures/confusion_full_M1_lower_half.png)
- [confusion_full_M2_lower_half.png](figures/confusion_full_M2_lower_half.png)
- [confusion_top12_M1_lower_half.png](figures/confusion_top12_M1_lower_half.png)
- [confusion_top12_M2_lower_half.png](figures/confusion_top12_M2_lower_half.png)
- [examples_correct_wrong.png](figures/examples_correct_wrong.png)
- [roc_M2.png](figures/roc_M2.png)
- [roc_mask_methods.png](figures/roc_mask_methods.png)
- [samples_per_class.png](figures/samples_per_class.png)
- [threshold_M2.png](figures/threshold_M2.png)
- [variants.png](figures/variants.png)
