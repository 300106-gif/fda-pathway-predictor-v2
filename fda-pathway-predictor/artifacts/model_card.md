# Model Card — FDA Pathway Predictor

## Purpose
Predicts FDA regulatory pathway (510(k) Exempt, 510(k), PMA, De Novo) for medical devices.
510(k) Exempt: low-risk Class I/II devices that may be marketed without premarket notification.

## Model: Gradient Boosting
## Features: 16
## Training Records: 53,500

## Metrics
| Metric | Value |
|---|---|
| Accuracy | 0.9804 |
| F1-macro | 0.8175 |
| CV F1 | 0.7953 ± 0.0037 |

## Limitations
- Trained on historical data; regulatory criteria can change
- Class imbalance (510(k) dominates)
- Does not analyze submission narratives or clinical evidence
- Decision support only — not regulatory advice

## Ethical Considerations
- May reflect historical biases in FDA decisions
- Must be used by qualified regulatory professionals
- All data is publicly available via openFDA API
