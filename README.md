# INNER-VOTE

Check whether a model agrees with itself: hidden-state consensus across paraphrases.

INNER-VOTE takes the states of N paraphrases of the same question at every layer and reports the mean and minimum pairwise cosine agreement per layer, the first layer where the paraphrases disagree, and the layer with the strongest internal consensus.

## What it does

- Layer-wise mean and minimum pairwise agreement across paraphrases.
- First-disagreement layer under a configurable threshold.
- Consensus peak layer and final agreement.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from inner_vote.vote import consensus_report

# states: (N paraphrases, D) per layer, or (N, T, D) for token states
layers = [np.load(f) for f in ["paraphrases_l0.npy", "paraphrases_l1.npy"]]
report = consensus_report(layers, low=0.7)
print(report["first_disagreement_layer"], report["peak_layer"])
```

## How it works

Paraphrase states are unit-normalized and compared pairwise by cosine. High agreement at a layer means the paraphrases share one internal representation there; a sudden drop marks the layer where surface differences start to matter inside the model.

## Honest scope

- Agreement measures representational similarity, not answer correctness.
- Paraphrase states must be width-aligned; token states are averaged first.
- Tests use synthetic states, so no model download is needed.

## License

MIT