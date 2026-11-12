# Results: fei

200 people, 2777 aligned faces. Test set: 70/30 stratified split of the images (seed 42).

## Accuracy (%) by test variant

| Variant | M0 Eigenfaces+SVM | M1 VGG-Face full | M2 VGG-Face full+partial | M3 FaceNet full | M4 FaceNet full+partial |
|---|---:|---:|---:|---:|---:|
| full (ใบหน้าเต็ม) | 52.6 | 99.0 | 98.7 | 98.2 | 98.0 |
| upper_half (ครึ่งบน) | 0.8 | 88.8 | 95.8 | 88.8 | 95.7 |
| lower_half (ครึ่งล่าง) | 0.8 | 17.9 | 86.7 | 42.7 | 87.4 |
| eyes_only (เฉพาะดวงตา) | 0.5 | 35.7 | 87.4 | 61.8 | 87.4 |
| nose_mouth (เฉพาะจมูก+ปาก) | 1.1 | 12.1 | 82.6 | 36.2 | 84.8 |
| mask (สวมหน้ากากอนามัย) | 2.0 | 83.3 | 95.3 | 86.9 | 95.3 |
| sunglasses (สวมแว่นกันแดด) | 5.5 | 49.9 | 94.6 | 76.7 | 94.5 |

## M2: all metrics

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Top-5 | Coverage@τ | Accuracy@τ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 0.987 | 0.990 | 0.987 | 0.987 | 1.000 | 0.999 | 0.964 | 0.995 |
| upper_half | 0.958 | 0.967 | 0.959 | 0.957 | 1.000 | 0.990 | 0.906 | 0.995 |
| lower_half | 0.867 | 0.890 | 0.867 | 0.862 | 0.997 | 0.947 | 0.676 | 0.991 |
| eyes_only | 0.874 | 0.914 | 0.874 | 0.877 | 0.997 | 0.938 | 0.697 | 0.998 |
| nose_mouth | 0.826 | 0.861 | 0.826 | 0.824 | 0.996 | 0.929 | 0.487 | 0.998 |
| mask | 0.953 | 0.964 | 0.954 | 0.952 | 1.000 | 0.988 | 0.887 | 0.997 |
| sunglasses | 0.946 | 0.958 | 0.946 | 0.945 | 0.999 | 0.989 | 0.841 | 0.999 |

## Targets (slide 5)

| | Target | Measured | |
|---|---|---|---|
| T1 Accuracy on full faces, M2 | ≥ 0.95 | 0.987 | PASS |
| T2 Accuracy with mask / sunglasses, M2 | ≥ 0.85 | mask 0.953, sunglasses 0.946 | PASS |
| T3 Acc(M2) - Acc(M1) on the variant where M1 is worst | +0.1 | nose_mouth +0.705 | PASS |

## Figures

- [accuracy_by_variant.png](figures/accuracy_by_variant.png)
- [augmentation.png](figures/augmentation.png)
- [class_distribution.png](figures/class_distribution.png)
- [confusion_full_M1_lower_half.png](figures/confusion_full_M1_lower_half.png)
- [confusion_full_M2_lower_half.png](figures/confusion_full_M2_lower_half.png)
- [confusion_top12_M1_lower_half.png](figures/confusion_top12_M1_lower_half.png)
- [confusion_top12_M2_lower_half.png](figures/confusion_top12_M2_lower_half.png)
- [examples_correct_wrong.png](figures/examples_correct_wrong.png)
- [pose_fei.png](figures/pose_fei.png)
- [roc_M2.png](figures/roc_M2.png)
- [roc_mask_methods.png](figures/roc_mask_methods.png)
- [samples_per_class.png](figures/samples_per_class.png)
- [threshold_M2.png](figures/threshold_M2.png)
- [variants.png](figures/variants.png)
