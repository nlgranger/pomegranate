#cython: boundscheck=False
#cython: cdivision=True
# normal.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.math cimport log as clog, sqrt as csqrt
from libc.math cimport M_PI as PI

import numpy as np
cimport numpy as np

from ..base cimport Model, DOUBLE_t, DOUBLE_t, INTP_t
from ..base import Model


cdef struct NormpdfSummaries:
    DOUBLE_t w_sum
    DOUBLE_t x_sum
    DOUBLE_t x2_sum


cdef class Normal(Model):
    """The univariate Normal (aka Gaussian) distribution $N(\mu, \sigma).$"""
    cdef public DOUBLE_t mu
    cdef DOUBLE_t two_sigma_square
    cdef NormpdfSummaries summaries

    @property
    def sigma(self):
        return (self.two_sigma_square / 2) ** .5

    @sigma.setter
    def sigma(self, value):
        self.two_sigma_square = 2 * (value ** 2)

    def __cinit__(self):
        self.summaries.w_sum = 0
        self.summaries.x_sum = 0
        self.summaries.x2_sum = 0
        Model.__init__(self, dshape=(1,), is_data_integral=False)

    def __init__(self, DOUBLE_t mu=0.0, DOUBLE_t sigma=1.0):
        """Return a Normal distribution of mean _mu_ and variance _sigma_.
        Defaults to $\mathcal{N}(0, 1)$.
        """

        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return "NormalDistribution({}, {})".format(self.mu, self.sigma)

    def get_params(self, deep=True):
        return {
            'is_frozen': self.is_frozen,
            'mu': self.mu,
            'sigma': self.sigma
        }

    def set_params(self, is_frozen, mu, sigma):
        self.__init__(mu, sigma)
        self.is_frozen = is_frozen

    cdef void log_probability_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] out):
        cdef DOUBLE_t a = - .5 * clog(PI * self.two_sigma_square)
        cdef DOUBLE_t b
        cdef int i

        for i in xrange(n):
            b = - ( (X[i] - self.mu) ** 2 ) / self.two_sigma_square
            out[i] = a + b

    def sample(self, n=None):
        return np.random.normal(self.mu, self.sigma, n)

    cdef void summarize_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] weights):
        cdef int i

        for i in xrange(n):
            self.summaries.w_sum += weights[i]
        for i in xrange(n):
            self.summaries.x_sum += weights[i] * X[i]
        for i in xrange(n):
            self.summaries.x2_sum += weights[i] * X[i] * X[i]

    cdef void from_summaries_fast(self, DOUBLE_t inertia):
        if self.is_frozen:
            return

        cdef DOUBLE_t x_expectation = \
            self.summaries.x_sum / self.summaries.w_sum
        cdef DOUBLE_t x2_expectation = \
            self.summaries.x2_sum / self.summaries.w_sum

        self.mu = inertia * self.mu + (1 - inertia) * x_expectation
        self.sigma = inertia * self.sigma + \
            (1 - inertia) * csqrt(x2_expectation - x_expectation ** 2)

        self.summaries.w_sum = 0
        self.summaries.x_sum = 0
        self.summaries.x2_sum = 0
