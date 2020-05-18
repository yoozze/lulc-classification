
import numpy as np


class SentinelHubValidData:
    """Combine Sen2Cor's classification map derived `IS_DATA` with
    SentinelHub's cloud mask `CLM`.
    """
    def __call__(self, eopatch):
        return np.logical_and(
            eopatch.mask['IS_DATA'].astype(np.bool),
            np.logical_not(eopatch.mask['CLM'].astype(np.bool))
        )


class ValidDataFractionPredicate:
    """Predicate that defines if a frame from EOPatch's time-series is valid or
    not. Frame is valid, if the valid data fraction is above the specified
    threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold
