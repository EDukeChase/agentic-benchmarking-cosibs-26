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
