# base.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy as np

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t INT_t
ctypedef np.intp_t INTP_t


cdef class Model:
    cdef public bint is_frozen
    cdef np.ndarray _dshape
    cdef bint _is_data_integral

    cdef void log_probability_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] out)

    cdef void log_probability_fast_i(self, np.ndarray[INT_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] out)

    cdef void summarize_fast(self, np.ndarray[DOUBLE_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] weights)

    cdef void summarize_fast_i(self, np.ndarray[INT_t, ndim=2] X,
            int n, np.ndarray[INTP_t, ndim=1] offsets,
            np.ndarray[DOUBLE_t, ndim=1] weights)

    cdef void from_summaries_fast(self, DOUBLE_t inertia)

