# Model Evaluation Report

**Date:** 2026-06-14
**Best Model:** Gradient Boosting

| Metric | Random Forest | Gradient Boosting |
|---|:---:|:---:|
| Accuracy | 0.9658 | 0.9804 |
| F1-macro | 0.7943 | 0.8175 |
| Precision | 0.7709 | 0.7951 |
| Recall | 0.8895 | 0.8632 |
| ROC-AUC | 0.9883 | 0.9866 |
| CV F1 | 0.7971 | 0.7953 |

## Classification Report — Gradient Boosting
```
              precision    recall  f1-score   support

        510k       1.00      0.98      0.99      9784
 510k_exempt       0.96      0.98      0.97       632
     De_Novo       0.23      0.49      0.31        82
         PMA       1.00      1.00      1.00       202

    accuracy                           0.98     10700
   macro avg       0.80      0.86      0.82     10700
weighted avg       0.99      0.98      0.98     10700
```
