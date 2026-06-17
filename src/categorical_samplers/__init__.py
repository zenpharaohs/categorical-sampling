"""Categorical-family random samplers."""

from .api import (
    BinomialStream,
    CategoricalStream,
    MultinomialStream,
    binomial,
    categorical,
    multinomial,
)
from .validation import (
    bhattacharyya_iid_bounds,
    empirical_affinity,
    randomized_categorical_pit,
    validate_categorical_draws,
)
from ._backend import native_available

__all__ = [
    "BinomialStream",
    "CategoricalStream",
    "MultinomialStream",
    "bhattacharyya_iid_bounds",
    "binomial",
    "categorical",
    "empirical_affinity",
    "multinomial",
    "native_available",
    "randomized_categorical_pit",
    "validate_categorical_draws",
]
