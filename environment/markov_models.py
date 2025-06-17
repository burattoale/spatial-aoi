import numpy as np

from utils import SimulationParameters
from utils import _compute_steady_state
from utils import cond_prob_y_given_x, cond_prob_y_given_x_spatial

class DTMC(object):
    def __init__(self, q, eta, seed=None):
        self._A = np.array([[1-q, q], [eta*q, 1-eta*q]])
        self._pi = _compute_steady_state(self._A)
        self.rng = np.random.default_rng(seed)
        self._state = int(self.rng.choice([0, 1], p=self._pi))

    @property
    def A(self) -> np.ndarray:
        return self._A
    
    @property
    def pi(self) -> np.ndarray:
        return self._pi
    
    @property
    def state(self):
        return self._state
    
    def step(self):
        next_state = int(self.rng.choice([0, 1], p=self.A[int(self.state)]))    
        self._state = next_state
        return next_state

    
class HiddenMM(object):
    """
    Contains the matrices and parameters of a Hidden Markov Model
    """

    def __init__(self, params:SimulationParameters, seed=None):
        self._params = params
        self.rng = np.random.default_rng(seed)
        self._A = np.array([[1 - self._params.q, self._params.q],
                            [self._params.eta * self._params.q, 1 - self._params.eta * self._params.q]])
        self._steady_state = _compute_steady_state(self._A)
        self._emission_matrix = self._fill_emission_matrix()
        
        # sample the initial state of the chain from steady state probabilities
        # the current visible state is drawn from the row of the emission matrix
        self._hidden_state = self.rng.choice(self._params.X_symbols, p=self._steady_state)
        self._emission_state = self.rng.choice(self._params.Y_symbols, p=self._emission_matrix[int(self._hidden_state)])

    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        return self._emission_matrix
    
    @property
    def pi(self):
        return self._steady_state
    
    @property
    def state(self):
        return self._emission_state
    
    @property
    def hidden_state(self):
        return self._hidden_state
    
    def step(self):
        next_hidden_state = self.rng.choice(self._params.X_symbols, p=self.A[int(self._hidden_state)])
        self._hidden_state = next_hidden_state
        self._emission_state = self.rng.choice(self._params.Y_symbols, p=self.B[int(self._hidden_state)])
        return self._emission_state
    
    def _fill_emission_matrix(self):
        shape = (len(self._params.X_symbols), len(self._params.Y_symbols))
        B = np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                B[i, j] = cond_prob_y_given_x(self._params.Y_symbols[j], 
                                              self._params.X_symbols[i], 
                                              self._params.zeta, 
                                              self._params.epsilon,
                                              self._params.m,
                                              self._params.K,
                                              self._params.alpha,
                                              self._params.R)

        return B
    

class SpatialHMM(HiddenMM):
    def __init__(self, params, seed=None):
        self._params = params
        self.rng = np.random.default_rng(seed)
        self._A = np.array([[1 - self._params.q, self._params.q],
                            [self._params.eta * self._params.q, 1 - self._params.eta * self._params.q]])
        self._steady_state = _compute_steady_state(self._A)
        
        # sample the initial state of the chain from steady state probabilities
        # the current visible state is drawn from the row of the emission matrix
        self._hidden_state = self.rng.choice(self._params.X_symbols, p=self._steady_state)
        self._emission_matrix = self._fill_emission_matrix()
        self._emission_state = self.rng.choice(self._params.Y_symbols, p=self._emission_matrix[int(self._hidden_state)])

    def _fill_emission_matrix(self):
        """
        Alternative formulation for the spatial features
        """
        shape = (len(self._params.X_symbols), len(self._params.Y_symbols))
        B = np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                B[i, j] = cond_prob_y_given_x_spatial(self._params.Y_symbols[j], 
                                              self._params.X_symbols[i], 
                                              self._params.zeta, 
                                              self._params.epsilon,
                                              self._params.m,
                                              self._params.K,
                                              self._params.alpha,
                                              self._params.R)

        return B