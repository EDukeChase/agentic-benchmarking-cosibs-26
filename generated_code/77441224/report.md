# Benchmark Report

## Overall Summary

CLMBR-T-base was selected because it is a published, explicitly documented structured-EHR foundation model aligned with few-shot prediction on clinical structured data. In this benchmark implementation, it was run as a decoder-only Transformer with causal local self-attention, using the public default-style configuration noted in the documentation, but with several explicit simplifications: inputs were assumed to be pre-tokenized standardized EHR codes, preprocessing/vocabulary construction were handled outside the module, the attention window was implemented as a strict causal local window, patient embeddings defaulted to the final valid token, and exact optimizer/training details were not reproduced. On the held-out EHR test set, it achieved accuracy 0.6657608695652174, precision 0.36363636363636365, recall 0.10526315789473684, f1 0.16326530612244897, auroc 0.5115002072109407, and brier 0.2687481641769409. The weak recall, low f1, and near-chance auroc are consistent with the documented limitations that this is not a byte-for-byte replica of the released checkpoint and that key preprocessing/training details were not fully specified; the modest precision suggests it made relatively few positive predictions, which likely contributed to the low recall.

## Recommendations

Based on the available evidence, CLMBR-T-base should not be prioritized for this task as implemented here. Although it has the strongest documentation pedigree among the provided details and an acceptable accuracy, the primary discrimination and positive-class performance metrics are poor: f1 0.16326530612244897, recall 0.10526315789473684, and auroc 0.5115002072109407 indicate performance close to chance. The implementation also relies on several explicit assumptions and omits exact preprocessing and training details, which reduces trustworthiness for deployment. If forced to choose from this benchmark alone, use CLMBR-T-base only as a baseline or reference implementation, not as the preferred predictive model. If additional candidate models are available, prefer the one with clearly better f1 and auroc, provided its implementation assumptions and limitations are similarly well documented.

## Model Results

| Model | Accuracy | F1 | Precision | Recall | AUROC | Brier | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CLMBR-T-base | 0.6657608695652174 | 0.16326530612244897 | 0.36363636363636365 | 0.10526315789473684 | 0.5115002072109407 | 0.2687481641769409 | success |

### CLMBR-T-base

#### Rationale

This is a strong benchmark candidate because it is a published, explicitly documented structured-EHR foundation model with enough architectural and training details to reproduce the general approach. It is also directly aligned with EHRSHOT’s goal of evaluating few-shot prediction on clinical structured data.

#### Implementation Notes

# CLMBR-T-base

## Implementation decisions
- Implemented as a decoder-only Transformer with **causal local self-attention**.
- Default configuration uses **12 layers**, **768 hidden units**, **12 heads**, **496-token local attention window**, and **0 dropout**, matching the public EHRSHOT/model-card descriptions.
- The module includes:
  - autoregressive next-token training loss,
  - representation extraction via final valid token or mean pooling,
  - a simple sequence classification head for downstream benchmarking,
  - checkpoint save/load helpers.

## Assumptions made
The public sources do not fully specify several low-level details, so the implementation makes the following explicit assumptions:
- Inputs are already tokenized as integer IDs for standardized EHR codes.
- Vocabulary construction, code normalization, and MEDS/FEMR preprocessing occur outside this module.
- The attention window is implemented as a strict causal local window over prior tokens.
- Patient embeddings are extracted from the **final valid token** by default.
- The model uses standard Transformer feed-forward blocks with GELU activations.
- Exact optimizer schedule, warmup, batch size, and training duration are not reproduced because they were not fully specified in the public sources reviewed.

## Known limitations
- This implementation does not reproduce the exact public CLMBR-T-base tokenizer/vocabulary.
- It does not include FEMR or MEDS preprocessing utilities.
- It is a clean research implementation intended for benchmarking, not a byte-for-byte replica of the released Stanford checkpoint.
- Memory usage of dense attention is acceptable for moderate sequence lengths, but large-batch pretraining may need optimized attention kernels.
