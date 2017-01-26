#cython: boundscheck=False
#cython: cdivision=True
# MixtureModel.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

from .base import Model
from .base cimport Model, DOUBLE_t
from .kmeans import Kmeans
from .utils cimport _log, pair_lse

DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class Mixture(Model):
    """A Mixture of distributions.

    Attributes
    ----------
    distributions: np.ndarray[Model]
        The components of the model.
    weight: np.ndarray[float]
        The mixture weights/proportions (must be assigned as a whole, not
        element-wise).
    """

    cdef public np.ndarray[object, ndim=1] distributions
    cdef np.ndarray[DOUBLE_t, ndim=1] log_weights
    cdef np.ndarray[DOUBLE_t, ndim=1] summaries

    @property
    def weights(self):
        return tuple(np.exp(self.log_weights))

    @weights.setter
    def weights(self, value):
        if len(value) != len(self.distributions):
            raise ValueError("invalid number of weights")
        self.log_weights = np.log(value) - np.log(np.sum(value))

    def __init__(self, distributions, weights=None):
        if len(distributions) < 2:
            raise ValueError("must have at least two distributions")
        if any(dis._is_vl != distributions[0].is_vl for dis in distributions) \
               or any(dis.d != distributions[0].d for dis in distributions):
            raise TypeError("mis-matching distribution dimensions")

        self.d = distributions[0].d
        self.is_vl = distributions[0].is_vl
        self.is_data_integral = distributions[0].is_data_integral
        self.is_fast_model = all(d.is_fast_model for d in distributions)
        self.distributions = distributions
        self.weights = [1.0] * len(distributions) if weights is None \
            else np.asarray(weights)
        self.summaries = np.zeros((len(distributions),), np.float64)

    def __repr__(self):
        return "<Mixture with {} components at {}>".format(
            len(self.components), id(self))

    def get_params(self, deep=True):
        return {
            'frozen': self.frozen,
            'distributions': self.distributions.tolist(),
            'weights': self.weights
        }

    def set_params(self, frozen, distributions, weights):
        self.__init__(distributions, weight)
        self.frozen = frozen

    cpdef log_probability(self, X):
        dtype = np.int32 if self.is_data_integral else np.float64
        packedX, n, offsets = data2array(X, self.is_vl, self.d, dtype)
        if self.is_fast_model:
            out = distributions[0].log_probability(X) + self.log_weights[0]
            self.log_probability_fast(X, n, offsets, out)
        else:
            out = np.empty((n,), dtype=np.float64)
            for i in xrange(1, len(self.distributions)):
                out = np.logaddexp(
                    out,
                    distributions[i].log_probability(X) + self.log_weights[i])

        return out

    def sample(self, n=1):
        samples = [self._distributions[d].sample()
                   for d in np.random.choice(self.n_components, self.weights)]
        return samples if n > 1 else samples[0]

    cdef void log_probability_fast(self, DOUBLE_t* X,
                                   int n, int* offsets,
                                   DOUBLE_t* log_probabilities) nogil:
        cdef int i, j
        cdef DOUBLE_t* tmp_log_probas = <DOUBLE_t*>malloc(n * sizeof(DOUBLE_t))

        for j in range(n):
            log_probabilities[j] = 0

        for i in range(self.k):
            (<Model> self.dist_ptrs[i]).log_probability_fast(
                X, n, offsets, tmp_log_probas)
            for j in range(n):
                tmp_log_probas[j] += self.log_weights[i]
                log_probabilities[j] = pair_lse(log_probabilities[j],
                                                tmp_log_probas[j])

        free(tmp_log_probas)


    # def fit(self, X, y=None, weights=None, inertia=0,
    #         stop_threshold=0.1, max_iterations=1e8, verbose=False):
    #     """Fit the model to new data using EM.
    #
    #     This method fits the components of the model to new data using the EM
    #     method. It will iterate until either max iterations has been reached,
    #     or the stop threshold has been passed.
    #
    #     This is a sklearn wrapper for train method.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_dimensions)
    #         A set of samples to train on.
    #
    #     y : array-like, shape (n_samples)
    #         Ignored
    #
    #     weights: array-like, shape (n_samples)
    #         Arbitrary positive scores which influence how samples influence
    #         the fitting process relatively to each other.
    #
    #     inertia : double
    #         When refitting, specifies what proportion of the current model is
    #         kept, ranging from 0.0 (ignore old value) to 1.0 (keep as is).
    #         Default is 0.0.
    #
    #     stop_threshold : double, optional, positive
    #         The threshold at which EM will terminate for the improvement of
    #         the model. If the model does not improve its fit of the data by
    #         a log probability of 0.1 then terminate.
    #         Default is 0.1.
    #
    #     max_iterations : int, optional, positive
    #         The maximum number of iterations to run EM for. If this limit is
    #         hit then it will terminate training, regardless of how well the
    #         model is improving per iteration.
    #         Default is 1e8.
    #
    #     verbose : bool, optional
    #         Whether or not to print out improvement information over
    #         iterations.
    #         Default is False.
    #
    #     Returns
    #     -------
    #     improvement : double
    #         The total improvement in log probability P(D|M)
    #     """
    #
    #     initial_log_probability_sum = NEGINF
    #     iteration, improvement = 0, INF
    #
    #     if weights is None:
    #         weights = np.ones(len(X), dtype=np.float64)
    #     else:
    #         weights = np.array(weights, dtype=np.float64)
    #
    #     while improvement > stop_threshold and iteration < max_iterations + 1:
    #         self.from_summaries(inertia)
    #         log_probability_sum = self.summarize(X, weights)
    #
    #         if iteration == 0:
    #             initial_log_probability_sum = log_probability_sum
    #         else:
    #             improvement = log_probability_sum - last_log_probability_sum
    #
    #             if verbose:
    #                 print("Improvement: {}".format(improvement))
    #
    #         iteration += 1
    #         last_log_probability_sum = log_probability_sum
    #
    #     self.clear_summaries()
    #
    #     if verbose:
    #         print("Total Improvement: {}".format(
    #             last_log_probability_sum - initial_log_probability_sum))
    #
    #     return last_log_probability_sum - initial_log_probability_sum
    #
    # cdef DOUBLE_t summarize_fast(self, DOUBLE_t[:, :] X, DOUBLE_t[:] weights,
    #                              int n, int[:] offsets) nogil:
    #
    # def summarize(self, X, weights=None):
    #     """Summarize a batch of data and store sufficient statistics.
    #
    #     This will run the expectation step of EM and store sufficient
    #     statistics in the appropriate distribution objects. The summarization
    #     can be thought of as a chunk of the E step, and the from_summaries
    #     method as the M step.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_dimensions)
    #         This is the data to train on. Each row is a sample, and each column
    #         is a dimension to train on.
    #
    #     weights : array-like, shape (n_samples,), optional
    #         The initial weights of each sample in the matrix. If nothing is
    #         passed in then each sample is assumed to be the same weight.
    #         Default is None.
    #
    #     Returns
    #     -------
    #     logp : double
    #         The log probability of the data given the current model. This is
    #         used to speed up EM.
    #     """
    #
    #     cdef int i, n, d
    #     cdef numpy.ndarray X_ndarray
    #     cdef numpy.ndarray weights_ndarray
    #     cdef double log_probability
    #
    #     if self.is_vl_:
    #         n, d = len(X), self.d
    #     elif self.d == 1:
    #         n, d = X.shape[0], 1
    #     elif self.d > 1 and X.ndim == 1:
    #         n, d = 1, len(X)
    #     else:
    #         n, d = X.shape
    #
    #     if weights is None:
    #         weights_ndarray = numpy.ones(n, dtype='float64')
    #     else:
    #         weights_ndarray = numpy.array(weights, dtype='float64')
    #
    #     # If not initialized then we need to do kmeans initialization.
    #     if self.d == 0:
    #         X_ndarray = _check_input(X, self.keymap)
    #         kmeans = Kmeans(self.n)
    #         kmeans.fit(X_ndarray, max_iterations=1)
    #         y = kmeans.predict(X_ndarray)
    #
    #         distributions = [
    #             self.distribution_callable.from_samples(X_ndarray[y==i])
    #             for i in range(self.n) ]
    #
    #         self.__init__(distributions)
    #
    #     cdef double* X_ptr
    #     cdef double* weights_ptr = <double*> weights_ndarray.data
    #
    #     if not self.is_vl_:
    #         X_ndarray = _check_input(X, self.keymap)
    #         X_ptr = <double*> X_ndarray.data
    #
    #         with nogil:
    #             log_probability = self._summarize(X_ptr, weights_ptr, n)
    #     else:
    #         log_probability = 0.0
    #         for i in range(n):
    #             X_ndarray = _check_input(X[i], self.keymap)
    #             X_ptr = <double*> X_ndarray.data
    #             d = len(X_ndarray)
    #             with nogil:
    #                 log_probability += self._summarize(X_ptr, weights_ptr+i, d)
    #
    #     return log_probability
    #
    # cdef double _summarize(self, double* X, double* weights, int n) nogil:
    #     cdef double* r = <double*> calloc(self.n*n, sizeof(double))
    #     cdef double* summaries = <double*> calloc(self.n, sizeof(double))
    #     cdef int i, j
    #     cdef double total, logp, log_probability_sum = 0.0
    #
    #     memset(summaries, 0, self.n*sizeof(double))
    #     cdef double tic
    #
    #     for j in range(self.n):
    #         if self.is_vl_:
    #             r[j*n] = (<Model> self.distributions_ptr[j]) \
    #                 ._vl_log_probability(X, n)
    #         else:
    #             (<Model> self.distributions_ptr[j]) \
    #                 ._v_log_probability(X, r+j*n, n)
    #
    #     for i in range(n):
    #         total = NEGINF
    #
    #         for j in range(self.n):
    #             r[j*n + i] += self.weights_ptr[j]
    #             total = pair_lse(total, r[j*n + i])
    #
    #         for j in range(self.n):
    #             r[j*n + i] = cexp(r[j*n + i] - total) * weights[i]
    #             summaries[j] += r[j*n + i]
    #
    #         log_probability_sum += total * weights[i]
    #
    #         if self.is_vl_:
    #             break
    #
    #     for j in range(self.n):
    #         (<Model> self.distributions_ptr[j])._summarize(X, r+j*n, n)
    #
    #     with gil:
    #         for j in range(self.n):
    #             self.summaries_ptr[j] += summaries[j]
    #
    #     free(r)
    #     free(summaries)
    #     return log_probability_sum
    #
    # def from_summaries(self, inertia=0.0, **kwargs):
    #     """Fit the model to the collected sufficient statistics.
    #
    #     Fit the parameters of the model to the sufficient statistics gathered
    #     during the summarize calls. This should return an exact update.
    #
    #     Parameters
    #     ----------
    #     inertia : double, optional
    #         The weight of the previous parameters of the model. The new
    #         parameters will roughly be
    #         old_param*inertia + new_param*(1-inertia),
    #         so an inertia of 0 means ignore the old parameters, whereas an
    #         inertia of 1 means ignore the new parameters. Default is 0.0.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #
    #     if self.d == 0 or self.summaries_ndarray.sum() == 0:
    #         return
    #
    #     self.summaries_ndarray /= self.summaries_ndarray.sum()
    #     for i, distribution in enumerate(self._distributions):
    #         distribution.from_summaries(inertia, **kwargs)
    #         self.weights[i] = _log(self.summaries_ndarray[i])
    #         self.summaries_ndarray[i] = 0.
    #
    # def predict_proba(self, X):
    #     """Calculate the posterior P(M|D) for data.
    #
    #     Calculate the probability of each item having been generated from
    #     each component in the model. This returns normalized probabilities
    #     such that each row should sum to 1.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_dimensions)
    #         The samples to do the prediction on. Each sample is a row and each
    #         column corresponds to a dimension in that sample. For univariate
    #         _distributions, a single array may be passed in.
    #
    #     Returns
    #     -------
    #     probability : array-like, shape (n_samples, n_components)
    #         The normalized probability P(M|D) for each sample. This is the
    #         probability that the sample was generated from each component.
    #     """
    #     return np.exp(self.predict_log_proba(X))
    #
    # def predict_log_proba(self, X):
    #     if self.d == 0:
    #         raise ValueError("must first fit model before using "
    #                  "predict_log_proba method.")
    #
    #     cdef int i, n, d
    #     cdef numpy.ndarray X_ndarray
    #     cdef double* X_ptr
    #     cdef numpy.ndarray y
    #     cdef double* y_ptr
    #
    #     if self.is_vl_:
    #         n, d = len(X), self.d
    #     elif self.d == 1:
    #         n, d = X.shape[0], 1
    #     elif self.d > 1 and X.ndim == 1:
    #         n, d = 1, len(X)
    #     else:
    #         n, d = X.shape
    #
    #     y = numpy.zeros((n, self.n), dtype='float64')
    #     y_ptr = <double*> y.data
    #
    #     if not self.is_vl_:
    #         X_ndarray = _check_input(X, self.keymap)
    #         X_ptr = <double*> X_ndarray.data
    #
    #     with nogil:
    #         if not self.is_vl_:
    #             self._predict_log_proba(X_ptr, y_ptr, n, d)
    #         else:
    #             for i in range(n):
    #                 with gil:
    #                     X_ndarray = _check_input(X[i], self.keymap)
    #                     X_ptr = <double*> X_ndarray.data
    #                     d = len(X_ndarray)
    #
    #                 self._predict_log_proba(X_ptr, y_ptr+i*self.n, 1, d)
    #
    #     return y if self.is_vl_ else y.reshape(self.n, n).T
    #
    # cdef void _predict_log_proba(self, double* X, double* y,
    #              int n, int d) nogil:
    #     cdef double y_sum, logp
    #     cdef int i, j
    #
    #     for j in range(self.n):
    #         if self.is_vl_:
    #             y[j] = (<Model> self.distributions_ptr[j]) \
    #                 ._vl_log_probability(X, d)
    #         else:
    #             (<Model> self.distributions_ptr[j]) \
    #                 ._v_log_probability(X, y+j*n, n)
    #
    #     for i in range(n):
    #         y_sum = NEGINF
    #
    #         for j in range(self.n):
    #             y[j*n + i] += self.weights_ptr[j]
    #             y_sum = pair_lse(y_sum, y[j*n + i])
    #
    #         for j in range(self.n):
    #             y[j*n + i] -= y_sum
    #
    # cpdef predict(self, X):
    #     """Predict the most likely component which generated each sample.
    #
    #     Calculate the posterior P(M|D) for each sample and return the index
    #     of the component most likely to fit it. This corresponds to a simple
    #     argmax over the responsibility matrix.
    #
    #     This is a sklearn wrapper for the maximum_a_posteriori method.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_dimensions)
    #         The samples to do the prediction on. Each sample is a row and each
    #         column corresponds to a dimension in that sample. For univariate
    #         _distributions, a single array may be passed in.
    #
    #     Returns
    #     -------
    #     y : array-like, shape (n_samples,)
    #         The predicted component which fits the sample the best.
    #     """
    #
    #     cdef int i, n, d
    #
    #     if self.d == 0:
    #         raise ValueError("must first fit model before using "
    #                  "predict method.")
    #
    #     if self.is_vl_:
    #         n, d = len(X), self.d
    #     elif self.d == 1:
    #         n, d = X.shape[0], 1
    #     elif self.d > 1 and X.ndim == 1:
    #         n, d = 1, len(X)
    #     else:
    #         n, d = X.shape
    #
    #     cdef numpy.ndarray X_ndarray
    #     cdef double* X_ptr
    #
    #     cdef numpy.ndarray y = numpy.zeros(n, dtype='int32')
    #     cdef int* y_ptr = <int*> y.data
    #
    #     if not self.is_vl_:
    #         X_ndarray = _check_input(X, self.keymap)
    #         X_ptr = <double*> X_ndarray.data
    #
    #     with nogil:
    #         if not self.is_vl_:
    #             self._predict(X_ptr, y_ptr, n, d)
    #         else:
    #             for i in range(n):
    #                 with gil:
    #                     X_ndarray = _check_input(X[i], self.keymap)
    #                     X_ptr = <double*> X_ndarray.data
    #                     d = len(X_ndarray)
    #
    #                 self._predict(X_ptr, y_ptr+i, 1, d)
    #
    #     return y
    #
    # cdef void _predict( self, double* X, int* y, int n, int d) nogil:
    #     cdef int i, j
    #     cdef double max_logp, logp
    #     cdef double* r = <double*> calloc(n*self.n, sizeof(double))
    #
    #     for j in range(self.n):
    #         if self.is_vl_:
    #             r[j] = (<Model> self.distributions_ptr[j]) \
    #                 ._vl_log_probability(X, d)
    #         else:
    #             (<Model> self.distributions_ptr[j]) \
    #                 ._v_log_probability(X, r+j*n, n)
    #
    #     for i in range(n):
    #         max_logp = NEGINF
    #
    #         for j in range(self.n):
    #             logp = r[j*n + i] + self.weights_ptr[j]
    #             if logp > max_logp:
    #                 max_logp = logp
    #                 y[i] = j
    #
    #     free(r)
    #
    #
    #
    # def clear_summaries(self):
    #     self.summaries_ndarray *= 0
    #     for distribution in self._distributions:
    #         distribution.clear_summaries()
    #
    # def to_json(self):
    #     separators=(',', ' : ')
    #     indent=4
    #
    #     model = {
    #                 'class' : 'GeneralMixtureModel',
    #                 '_distributions'  : [ json.loads(dist.to_json())
    #                          for dist in self._distributions ],
    #                 'weights' : self.weights.tolist()
    #             }
    #
    #     return json.dumps(model, separators=separators, indent=indent)
    #
    # @classmethod
    # def from_json(cls, s):
    #     d = json.loads(s)
    #     distributions = [ Distribution.from_json(json.dumps(j))
    #               for j in d['_distributions'] ]
    #     model = GeneralMixtureModel(distributions, numpy.array( d['weights'] ))
    #     return model
