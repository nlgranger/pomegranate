# base.pyx
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )

import json
cimport numpy as np
import numpy as np
import importlib

from .utils cimport *


DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef class Model(object):
    """Base class for all _distributions.

    Attributes
    ----------
    d : int
        The dimensionality of the flattened data. Example: for a 2 by 3
        dimensional space, d is 6. For variable length observations,
        this is the dimensionality of a single item.

    is_vl: boolean
        Indicates wether this model takes variable length inputs.

    is_frozen: boolean
        When true, calls to :func:`fit` and :func:`from_summaries` are
        silently ignored.
    """

    def __cinit__(self):
        self.d = 0
        self.is_vl = False
        self.is_frozen = False

    def __reduce__(self):
        return self.__class__, tuple(), self.get_params()

    def __setstate__(self, state):
        self.set_params(**state)

    def get_params(self, deep=True):
        """Return the parameters and state of this model."""
        # TODO: document that this is used by default in __reduce__ and to_json
        raise NotImplementedError

    def set_params(self, **params):
        """Set the parameters and the state of this Model from the result of
        :func:`Model.get_params`.
        """

        raise NotImplementedError

    def freeze(self):
        """Freeze the distribution, preventing updates from occuring."""
        self.is_frozen = True

    def thaw(self):
        """Thaw the distribution, re-allowing updates to occur."""
        self.is_frozen = False

    def log_probability(self, X):
        """Return the log-probabilities of symbols."""

        X, n, offsets = self._data2array(X, dtype=np.float64)
        res = np.empty((n,), dtype=np.float64)

        cdef DOUBLE_t* X_ = <DOUBLE_t*> &X[0]
        cdef int n_ = n
        cdef int* offsets_ = <int*> &offsets[0]
        cdef DOUBLE_t* res_ = <DOUBLE_t*> res.data

        with nogil:
            self.log_probability_fast(X_, n_, offsets_, res_)

        return res

    cdef void log_probability_fast(self, DOUBLE_t* X,
                                   int n, int* offsets,
                                   DOUBLE_t* log_probabilities) nogil:
        pass

    def probability(self, symbols):
        """Return the probabilities of symbols."""
        return np.exp(self.log_probability(symbols))

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

        self.summarize(X.data, weights.data)
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

        Returns
        -------
        None
        """

        if self.is_frozen:
            return
        if len(X) == 0:
            raise ValueError("sample is empty")

        X, n, offsets = self._data2array(X)
        weights = self._weights2array(weights, n)

        cdef DOUBLE_t* X_ = <DOUBLE_t*> X.data
        cdef DOUBLE_t* weights_ = <DOUBLE_t*> weights.data
        cdef int n_ = n
        cdef int* offsets_ = <int*> offsets.data

        with nogil:
            self.summarize_fast(X_, weights_, n_, offsets_)

    cdef void summarize_fast(self, DOUBLE_t* X, DOUBLE_t* weights,
                             int n, int* offsets) nogil:
        pass

    def from_summaries(self, inertia=0.0):
        """Fit the distribution to stored sufficient statistics following
        calls to :ref:`Model.summarize`. Summaries will then be reset.

        Parameters
        ----------
        inertia : double, optional
            When refitting, specifies what proportion of the current model is
            kept, ranging from 0.0 (ignore old state) to 1.0 (keep as is).
            Default is 0.0.

        Returns
        -------
        None
        """

        return NotImplementedError

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

    def to_json(self):
        """Serialize this object to JSON format.

        Returns
        -------
        model : str
            A JSON representation of this object instance.

        See :func:`from_json` for implementation details.
        """

        return json.dumps({
            'class' : self.__class__.__module__ +
                      "." + self.__class__.__name__,
            'frozen' : self.is_frozen,
            'state' : self.get_params()
        })

    @classmethod
    def from_json(cls, s):
        """Deserialize a model from its JSON representation.

        Parameters
        ----------
        s : str
            A JSON model state as returned by :ref:`to_json`

        Returns
        -------
        model : object
            A properly instanciated distribution.

        # <<<<<---- TODO: move this section to internal API documentation
        Note
        ----
        The deserialized JSON data should be a dictionary containing at least
        a 'class' key with a string item of the form:
        _importable_module_._ClassName_.

        This helps the default implementation of :func:`Model.from_json`
        forward the JSON representation to the appropriate class for any type
        of model.

        Example
        -------
        >>> import pomegranate
        >>>
        >>> s = { 'class': 'pomegranate.UniformDistribution',
        >>>       'start': 0.0,
        >>>       'stop': 1.0 }
        >>> d = pomegranate.Model.from_json(s)
        >>>
        >>> print(isinstance(d, pomegranate.UniformDistribution))
        True
        >>> print(d.start)
        0.0
        >>> print(d.stop)
        1.0
        """

        d = json.loads(s)
        if 'class' not in d.keys():
            raise ValueError("missing 'class' field")

        if cls == Model:  # Explicitely called implementation
            path = d['class'].split('.')

            distmodule = importlib.import_module('.'.join(path[:-1]))
            distclass = getattr(distmodule, path[-1])
            return distclass.from_json(s)

        else:  # Default inherited implementation
            obj = cls()
            obj.set_params(**(d['state']))
            obj.is_frozen = d['frozen']
            return obj

    def _data2array(self, X, weights=None, dtype=np.float64):
        # Concatenate samples into a single array for fast access
        
        if self.is_vl:
            durations = np.array([len(x) for x in X], dtype=np.int)
            offsets = np.cumsum(durations)
        
            data = np.empty((np.sum(durations), self.d), dtype=dtype)
            o = 0
            for i, (x, d) in enumerate(zip(X, durations)):
                data[o:o + d, :] = x.reshape((-1, self.d))
                o = o + d
            
            return X, len(X), offsets
        
        else:
            X = np.asarray(X)
            if len(X.shape) != 2 + self.is_vl:
                X = X.reshape((-1, self.d))

            if X.dtype != dtype:
                X = X.astype(dtype)
            X = X.reshape((-1, self.d))
            return X, len(X), np.zeros((0,), dtype=np.int32)
    
    @staticmethod
    def _weights2array(W, n, dtype=np.float64):
        if W is None:  # weight everything 1
            return np.ones((n,), dtype=dtype)
        else:  # force whatever we have to be a Numpy array
            return np.array(W, dtype=dtype)


# cdef class GraphModel(Model):
#     """Base class for graphical models."""
#
#     def __init__(self, name=None):
#         self.nodes = []
#         self.edges = []
#         self.d = 0
#
#     def add_node(self, *node):
#         """Add one or several nodes to the graph."""
#         for n in node:
#             self.nodes.append(n)
#
#     def add_state(self, *state):
#         """Alias for :func:`add_node`."""
#         self.add_node(*state)
#
#     def add_edge(self, a, b):
#         """Add an edge between two nodes/states in the graph.
#
#         For oriented graphs: the edge is oriented from a to b.
#         """
#
#         self.edges.append( (a, b) )
#
#     def add_transition(self, a, b):
#         """Alias for :func:`add_edge`."""
#         self.add_edge(a, b)
#
#     def get_params(self, deep=True):
#         raise NotImplementedError
#
#     def set_params(self, **params):
#         raise NotImplementedError
#
#     def log_probability(self, X):
#         raise NotImplementedError
#
#     def fit(self, X, y, weights=None, inertia=0, **kwargs):
#         raise NotImplementedError
#
#     def summarize(self, X, weights=None):
#         return NotImplementedError
#
#     def from_summaries(self, inertia=0.0):
#         return NotImplementedError
#
#     def sample(self, n=None):
#         raise NotImplementedError
#
#     # def dense_transition_matrix(self):
#     #     """Return the dense transition matrix.
#     #
#     #     Useful if the transitions of somewhat small models need to be analyzed.
#     #     """
#     #
#     #     m = len(self.nodes)
#     #     transition_log_probabilities = numpy.zeros( (m, m) ) + NEGINF
#     #
#     #     for i in range(m):
#     #         for n in range( self.out_edge_count[i], self.out_edge_count[i+1] ):
#     #             transition_log_probabilities[i, self.out_transitions[n]] = \
#     #                 self.out_transition_log_probabilities[n]
#     #
#     #     return transition_log_probabilities
#
#
# cdef class State(object):
#     """Represents a state in an HMM. Holds emission distribution, but not
#     transition distribution, because that's stored in the graph edges.
#     """
#
#     def __init__(self, distribution, name=None, weight=None):
#         """
#         Make a new State emitting from the given distribution. If distribution
#         is None, this state does not emit anything. A name, if specified, will
#         be the state's name when presented in output. Name may not contain
#         spaces or newlines, and must be unique within a model.
#         """
#
#         # Save the distribution
#         self.distribution = distribution
#
#         # Save the name
#         self.name = name or str(uuid.uuid4())
#
#         # Save the weight, or default to the unit weight
#         self.weight = weight or 1.
#
#     def __reduce__(self):
#         return self.__class__, (self.distribution, self.name, self.weight)
#
#     def tie( self, state ):
#         """
#         Tie this state to another state by just setting the distribution of the
#         other state to point to this states distribution.
#         """
#         state.distribution = self.distribution
#
#     def is_silent(self):
#         """
#         Return True if this state is silent (distribution is None) and False
#         otherwise.
#         """
#         return self.distribution is None
#
#     def tied_copy(self):
#         """
#         Return a copy of this state where the distribution is tied to the
#         distribution of this state.
#         """
#         return State( distribution=self.distribution, name=self.name+'-tied' )
#
#     def copy( self ):
#         """Return a hard copy of this state."""
#         return State( distribution=self.distribution.copy(), name=self.name )
#
#     def to_json(self):
#         """Convert this state to JSON format."""
#
#         return json.dumps({
#             'class' : self.__class__.__module__ \
#                       + '.' + self.__class__.__name__,
#             'distribution' : None if self.is_silent()
#                              else json.loads( self.distribution.to_json() ),
#             'name' : self.name,
#             'weight' : self.weight
#             })
#
#     @classmethod
#     def from_json( cls, s ):
#         """Read a State from a given string formatted in JSON."""
#
#         # Load a dictionary from a JSON formatted string
#         d = json.loads(s)
#
#         # If we're not decoding a state, we're decoding the wrong thing
#         if d['class'] != 'State':
#             raise IOError( "State object attempting to decode "
#                            "{} object".format( d['class'] ) )
#
#         # If this is a silent state, don't decode the distribution
#         if d['distribution'] is None:
#             return cls( None, str(d['name']), d['weight'] )
#
#         # Otherwise it has a distribution, so decode that
#         name = str(d['name'])
#         weight = d['weight']
#
#         c = d['distribution']['class']
#         dist = eval(c).from_json( json.dumps( d['distribution'] ) )
#         return cls( dist, name, weight )