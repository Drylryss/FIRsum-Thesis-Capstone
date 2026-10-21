"""Class weights are estimated only from training labels, never validation or test."""
from collections import Counter
import math
from core import LABELS
import torch
from transformers import Trainer


def oversample_issues(rows, factor=1):
    """Repeat training ISSUE rows only; never modify source annotations."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('ISSUE oversampling factor must be a positive integer')
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('Oversampling accepts training rows only')
    return [dict(r) for r in rows for _ in range(factor if r['label'] == 'ISSUE' else 1)]


def class_weights(rows, mode='none', cap=20.0, issue_multiplier=1.0):
    if mode not in {'none', 'balanced'} or cap <= 0 or issue_multiplier <= 0:
        raise ValueError('Choose none/balanced with positive cap and ISSUE multiplier')
    if not math.isfinite(cap) or not math.isfinite(issue_multiplier):
        raise ValueError('Weights must be finite')
    counts = Counter(r['label'] for r in rows)
    if any(counts[r] == 0 for r in LABELS):
        raise ValueError('All four classes must be present in training')
    weights = [len(rows) / (len(LABELS) * counts[r]) if mode == 'balanced' else 1.0 for r in LABELS]
    weights[LABELS.index('ISSUE')] *= issue_multiplier
    return [min(cap, w) for w in weights]


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, loss_normalizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_normalizer = loss_normalizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
        weights = self.class_weights.to(device=outputs.logits.device, dtype=outputs.logits.dtype)
        if getattr(self, 'loss_normalizer', None) is None:
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
        else:
            # Fixed training-set denominator preserves weighting across minibatches.
            # The default weighted CE divides by each batch's own summed weights.
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights,
                reduction='none').mean() / self.loss_normalizer
        return (loss, outputs) if return_outputs else loss
