# Model Evaluation Report

**Date:** 2026-06-14
**Best Model:** Random Forest

| Metric | Random Forest | Gradient Boosting |
|---|:---:|:---:|
| Accuracy | 0.9755 | 0.9836 |
| F1-macro | 0.7612 | 0.7611 |
| Precision | 0.7303 | 0.7405 |
| Recall | 0.8681 | 0.7993 |
| ROC-AUC | 0.9829 | 0.9801 |
| CV F1 | 0.7575 | 0.7479 |

## Classification Report — Random Forest
```
              precision    recall  f1-score   support

        510k       1.00      0.98      0.99      9784
     De_Novo       0.19      0.63      0.30        83
         PMA       1.00      1.00      1.00       202

    accuracy                           0.98     10069
   macro avg       0.73      0.87      0.76     10069
weighted avg       0.99      0.98      0.98     10069
```
