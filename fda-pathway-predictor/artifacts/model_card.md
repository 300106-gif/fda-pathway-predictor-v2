# Model Card — FDA Pathway Predictor

## Purpose
Predicts FDA regulatory pathway (510(k), PMA, De Novo) for medical devices.

## Model: Random Forest
## Features: 15
## Training Records: 50,342

## Metrics
| Metric | Value |
|---|---|
| Accuracy | 0.9755 |
| F1-macro | 0.7612 |
| CV F1 | 0.7575 ± 0.0050 |

## Limitations
- Trained on historical data; regulatory criteria can change
- Class imbalance (510(k) dominates)
- Does not analyze submission narratives or clinical evidence
- Decision support only — not regulatory advice

## Ethical Considerations
- May reflect historical biases in FDA decisions
- Must be used by qualified regulatory professionals
- All data is publicly available via openFDA API
