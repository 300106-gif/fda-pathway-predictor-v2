# Model Card — FDA Pathway Predictor

## Purpose
Predicts FDA regulatory pathway (510(k), PMA, De Novo) for medical devices.

## Model: Random Forest
## Features: 19
## Training Records: 10,013

## Metrics
| Metric | Value |
|---|---|
| Accuracy | 1.0000 |
| F1-macro | 1.0000 |
| CV F1 | 1.0000 ± 0.0000 |

## Limitations
- Trained on historical data; regulatory criteria can change
- Class imbalance (510(k) dominates)
- Does not analyze submission narratives or clinical evidence
- Decision support only — not regulatory advice

## Ethical Considerations
- May reflect historical biases in FDA decisions
- Must be used by qualified regulatory professionals
- All data is publicly available via openFDA API
