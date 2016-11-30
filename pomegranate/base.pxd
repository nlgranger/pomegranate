# base.pxd
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

cimport numpy as np

ctypedef np.float64_t DOUBLE_t
# ctypedef np.intp_t INTP_t
# # ctypedef fused SYMBOL_t:
# # 	DOUBLE_t
# # 	INTP_t
# ctypedef DOUBLE_t SYMBOL_t
# ctypedef np.intp_t LABEL_t


cdef class Model(object):
	cdef public int d
	cdef public bint is_frozen
	cdef public bint is_vl

	cdef void log_probability_fast(self, DOUBLE_t[:, :] symbols,
	                               int n, int[:] offsets,
	                               DOUBLE_t[:] log_probabilities) nogil

	cdef void summarize_fast(self, DOUBLE_t[:, :] X, DOUBLE_t[:] weights,
	                         int n, int[:] offsets) nogil


cdef class GraphModel(Model):
	cdef public list states, edges
	cdef public object graph
	cdef int n_edges, n_states


cdef class State(object):
	cdef public Model distribution
	cdef public str name
	cdef public DOUBLE_t weight
