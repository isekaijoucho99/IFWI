"""Interleaved source groups with a private cyclic group-label shift.

For C[b,s] supported at b=s%B, a row roll by k gives b=(s+k)%B.
This only permutes group labels: C.T@C and all shot co-memberships are
unchanged. It is not a random partition and does not remove within-group
cross terms. Fresh Rademacher signs still come from the existing encoder.
"""
from __future__ import annotations

import numpy as np
import torch

from core_speed import SourceEncoder
from experiments.dense_encoding_physics import DenseAcquisition, _integer


class _ShiftedSourceEncoder(SourceEncoder):
    def shift(self, update):
        update = _integer(update, 'update', 0)
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed % 2**32, update % 2**32, 0x53484946]))
        return int(rng.integers(self.supershots))

    def encoding_matrix(self, update):
        # Validate before the existing problem() can assign a code to a solver.
        shift = self.shift(update)
        matrix = super().encoding_matrix(update)
        return torch.roll(matrix, shifts=shift, dims=0)


class ShiftedInterleavedAcquisition(DenseAcquisition):
    """Same batched solver and normalization, with cyclic row relabeling.

    The encoder supplies the shifted code before SourceEncoder.problem sets
    the ordinary per-update solver weights and creates encoded observations.
    Thus solver weights, returned C and target use exactly the same code;
    no solver/geometry replacement or post-hoc target-only roll is needed.
    """
    encoder_type = _ShiftedSourceEncoder

    def __init__(self, problem, batch_size=13, fields=2, seed=3):
        super().__init__(problem,batch_size=batch_size,grouping='interleaved',
                         fields=fields,seed=seed)

    def shift(self, update):
        return self.encoders[0].shift(update)

    def batches(self, update):
        for batch in super().batches(update):
            # The legacy group-size formula precedes row relabeling. Recount
            # from C to cover unequal groups as well as the N52/B2/B4 cases.
            yield {**batch,'encoding_shift':self.shift(update),
                'encoding_membership_changed':False,
                'group_sizes':batch['encoding_matrix'].ne(0).sum(dim=1).tolist()}
