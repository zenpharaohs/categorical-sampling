"""Categorical-family random samplers."""

from .api import (
    BinomialStream,
    CategoricalStream,
    MultinomialStream,
    binomial,
    categorical,
    multinomial,
)
from .validation import bhattacharyya_iid_bounds, empirical_affinity

__all__ = [
    "BinomialStream",
    "CategoricalStream",
    "MultinomialStream",
    "bhattacharyya_iid_bounds",
    "binomial",
    "categorical",
    "empirical_affinity",
    "multinomial",
]
