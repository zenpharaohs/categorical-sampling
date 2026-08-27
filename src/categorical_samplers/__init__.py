"""Categorical-family random samplers."""

__version__ = "0.1.0a1"

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
from .tuning import MultinomialContext, MultinomialTuner

__all__ = [
    "BinomialStream",
    "CategoricalStream",
    "MultinomialStream",
    "MultinomialContext",
    "MultinomialTuner",
    "bhattacharyya_iid_bounds",
    "binomial",
    "categorical",
    "empirical_affinity",
    "multinomial",
    "native_available",
    "randomized_categorical_pit",
    "validate_categorical_draws",
    "__version__",
]
