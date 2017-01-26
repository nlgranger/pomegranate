#cython: boundscheck=False
#cython: cdivision=True
# uniform.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

from libc.math cimport log as clog
import numpy as np
cimport numpy as np

from ..base import Model
from ..base cimport Model, DOUBLE_t, INTP_t


# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class UniformReal(Model):
    """A uniform real distribution between two boundaries."""

    cdef public DOUBLE_t start
    cdef public DOUBLE_t stop
    cdef DOUBLE_t summaries_start
    cdef DOUBLE_t summaries_stop

    def __cinit__(self):
        self.summaries_start = INF
        self.summaries_stop = NEGINF
        Model.__init__(self, dshape=(1,), is_data_integral=False)

    def __init__(self, DOUBLE_t start=0, DOUBLE_t stop=1):
        """Return a Uniform real distribution spanning between start and
        end inclusive.
        Defaults to $\mathcal{U}([0, 1])$.
        """
        self.start = min(start, stop)
        self.stop = max(start, stop)

    def __repr__(self):
        return "UniformDistribution({}, {})".format(self.start, self.stop)

    def get_params(self, deep=True):
        return {
            'is_frozen': self.is_frozen,
            'start': self.start,
            'stop': self.stop
        }

    def set_params(self, is_frozen, start, stop):
        self.__init__(start, stop)
        self.is_frozen = is_frozen

    def sample(self, n=None):
        return np.random.uniform(self.start, self.stop, n)

    cdef void log_probability_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] out):
        cdef DOUBLE_t logp = - clog(self.stop - self.start)
        cdef int i

        for i in xrange(n):
            if self.start <= X[i] <= self.stop:
                out[i] = logp
            else:
                out[i] = NEGINF

    cdef void summarize_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] weights):
        cdef int i

        for i in xrange(n):
            if weights[i] == 0:
                continue
            elif X[i] < self.summaries_start:
                self.summaries_start = X[i]
            elif X[i] > self.summaries_stop:
                self.summaries_stop = X[i]

    cdef void from_summaries_fast(self, DOUBLE_t inertia):
        if self.is_frozen:
            return

        self.start = self.start * inertia + (1-inertia) * self.summaries_start
        self.stop = self.start * inertia + (1-inertia) * self.summaries_stop

        self.summaries_start = INF
        self.summaries_stop = NEGINF
