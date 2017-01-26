# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import json
import numpy as np
cimport numpy as np

from .utils import data2array, weights2array


cdef class Model:
    """Base class for all _distributions.

    Attributes
    ----------

    is_frozen: boolean
        When set to `True`, the current parameters are locked so that
        calls to :func:`fit` and :func:`from_summaries` are silently ignored.
    """

    def __cinit__(self):
        self.is_frozen = False
        self._dshape = np.empty((0,), dtype=int)

    def __init__(self, dshape, is_data_integral):
        new_shape = np.array(dshape, dtype=int)
        if not (len(new_shape.shape) == 1 and len(new_shape) > 0) \
                or not np.all(new_shape[1:] > 0) \
                or not (new_shape[0] > 0 or new_shape[0] == -1):
            raise ValueError("invalid shape specification")

        self._dshape = new_shape
        self._is_data_integral = is_data_integral

    @property
    def dshape(self):
        return tuple(self._dshape)

    @property
    def dtype(self):
        return np.int32 if self._is_data_integral else np.float64

    def __reduce__(self):
        return self.__class__, tuple(), self.get_params()

    def __setstate__(self, state):
        self.set_params(**state)

    def get_params(self, deep=True):
        """Return the parameters and state of this model."""
        # TODO: document that this is used by __reduce__ and to_json
        # TODO: document that devs should not forget is_frozen
        raise NotImplementedError

    def set_params(self, **params):
        """Set the parameters and the state of this Model from the result of
        :func:`Model.get_params`.
        """
        raise NotImplementedError

    def log_probability(self, X):
        """Return the log-probabilities of symbols."""

        packedX, n, offsets = data2array(X, self.dshape, self.dtype)
        out = np.empty((n,), dtype=np.float64)
        if isinstance(self.dtype(1), float):
            self.log_probability_fast(packedX, n, offsets, out)
        else:
            self.log_probability_fast_i(packedX, n, offsets, out)

        if out[0] > 0: # default implementation gives invalid value on purpose
            raise NotImplementedError("either log_probability or "
                "log_probability_fast must be implemented")

        return out

    def probability(self, X):
        """Return the probabilities of symbols."""
        return np.exp(self.log_probability(X))

    def fit(self, X, y=None, weights=None, inertia=0, **kwargs):
        """Fit the distribution to new data using MLE estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            A set of samples to train on.

        y : array-like, shape (n_samples)
            If relevant to the model, labels associated to the samples.

        weights: array-like, shape (n_samples)
            Arbitrary positive scores which influence how samples influence
            the fitting process relatively to each other.

        inertia : double
            When refitting, specifies what proportion of the current model is
            kept, ranging from 0.0 (ignore old value) to 1.0 (keep as is).
            Default is 0.0.

        Returns
        -------
        The current (fitted) distribution
        """

        if self.is_frozen or inertia == 1.0:
            return self

        self.summarize(X, weights)
        self.from_summaries(inertia)
        return self

    def summarize(self, X, weights=None):
        """Summarize a batch of data into sufficient statistics for a later
        update. If called several times without triggering updates,
        the statistics are stacked together so that a final call to
        :ref:`Model.from_summarize` applies the update for all the batches.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            A set of samples to train on.

        weights: array-like, shape (n_samples)
            Arbitrary positive scores which influence how samples influence
            the fitting process relatively to each other.
        """

        packedX, n, offsets = data2array(X, self.dshape, self.dtype)
        weights = weights2array(weights, n)

        if isinstance(self.dtype(1), float):
            self.summarize_fast(packedX, n, offsets, weights)
        else:
            self.summarize_fast_i(packedX, n, offsets, weights)

    def from_summaries(self, inertia=0.0):
        """Fit the distribution to stored sufficient statistics following
        calls to :ref:`Model.summarize`. Summaries will then be reset.

        Parameters
        ----------
        inertia : double, optional
            When refitting, specifies what proportion of the current model is
            kept, ranging from 0.0 (ignore old state) to 1.0 (keep as is).
            Default is 0.0.
        """

        self.from_summaries_fast(inertia)

    def sample(self, n=None):
        """Generate random samples from this distribution.

        Parameters
        ----------
        n : int or None, optional
            The number of samples to return. Default is None, to generate a
            single sample.

        Returns
        -------
        sample(s):
            One sample if _n_ is `None`, otherwise an array of _n_ samples.
        """

        raise NotImplementedError

    cdef void log_probability_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] out):

        if self.__class__.log_probability == Model.log_probability:
            # log_probability is not implemented, and neither is this method,
            # let's return an invalid value
            out[0] = 1

        if self._dshape[0] == -1:
            out[:] = self.log_probability(
                [X[offsets[i]:offsets[i+1]].reshape(self.dshape)
                 for i in xrange(n)])
        else:
            out[:] = self.log_probability(X.reshape((-1,) + self.dshape))

    cdef void log_probability_fast_i(self, np.ndarray[INT_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] out):

        if self.__class__.log_probability == Model.log_probability:
            # log_probability is not implemented, and neither is this method,
            # let's return an invalid value
            out[0] = 1

        if self._dshape[0] == -1:
            out[:] = self.log_probability(
                [X[offsets[i]:offsets[i+1], :].reshape(self.dshape)
                 for i in xrange(n)])
        else:
            out[:] = self.log_probability(X.reshape((-1,) + self.dshape))

    cdef void summarize_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] weights):

        if self.__class__.summarize == Model.summarize:
            return  # TODO: fail gracefully

        if self._dshape[0] == -1:
            self.summarize(
                [X[offsets[i]:offsets[i+1]].reshape(self.dshape)
                 for i in xrange(n)], weights)
        else:
            self.summarize(X.reshape((-1,) + self.dshape), weights)

    cdef void summarize_fast_i(self, np.ndarray[INT_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] weights):

        if self.__class__.summarize == Model.summarize:
            return  # TODO: fail gracefully

        if self._dshape[0] == -1:
            self.summarize(
                [X[offsets[i]:offsets[i+1]].reshape(self.dshape)
                 for i in xrange(n)], weights)
        else:
            self.summarize(X.reshape((-1,) + self.dshape), weights)

    cdef void from_summaries_fast(self, DOUBLE_t inertia):
        pass  # TODO: fail gracefully
