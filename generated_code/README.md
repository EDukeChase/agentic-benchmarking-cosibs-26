# EHRSHOT ML Benchmarks

Modular PyTorch and tabular baselines for structured EHR benchmark tasks inspired by EHRSHOT.

Implemented models:
- CLMBR-T-base-like causal Transformer encoder
- XGBoost on count-based features
- Random Forest on count-based features
- GRU longitudinal model
- LSTM longitudinal model

## Notes

- The CLMBR-T-base implementation here is a reproducible, local approximation of the published architecture and training objective.
- Exact EHRSHOT/FEMR preprocessing, ontology expansion, and published pretrained weights are not bundled here.
- Assumptions that remain project-specific: tokenization, visit windowing, ontology expansion, and split protocol.
