import numpy as np

from utils import SimulationParameters
from utils import _compute_steady_state
from utils import cond_prob_y_given_x, cond_prob_y_given_x_spatial, cond_prob_y_given_x_non_binary
from utils import PoiBin
from utils import lam

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
        assert len(self._params.X_symbols) <= len(self._params.Y_symbols)
        self.lambda_mat = self._generate_lamda_matrix()
        self.rng = np.random.default_rng(seed)
        self._A = np.array([[1 - self._params.q, self._params.q],
                            [self._params.eta * self._params.q, 1 - self._params.eta * self._params.q]])
        if len(self._params.X_symbols) > 2:
            self._A = self._build_tranmission_matrix()
            print(self._A)
        self._steady_state = _compute_steady_state(self._A)
        self._emission_matrix = self._fill_emission_matrix()
        self._emission_matrix = np.clip(self._emission_matrix, 0, 1)
        
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
        p_succ = None
        p_idle = None
        flag=False
        states_cardinality = shape[0]
        if isinstance(self._params.zeta, list) or isinstance(self._params.zeta, np.ndarray):
            poibin = PoiBin(self._params.zeta)
            p_succ = poibin.pmf(1)
            p_idle = poibin.pmf(0)
            flag = True
        for i in range(shape[0]):
            for j in range(shape[1]):
                if flag:
                    B[i, j] = cond_prob_y_given_x(self._params.Y_symbols[j], 
                                              self._params.X_symbols[i], 
                                              self._params.zeta, 
                                              self._params.epsilon,
                                              self._params.m,
                                              self._params.K,
                                              self._params.alpha,
                                              self._params.R,
                                              poibin=True,
                                              p_succ=p_succ,
                                              p_idle=p_idle)
                elif states_cardinality > 2:
                    B[i, j] = cond_prob_y_given_x_non_binary(self._params.Y_symbols[j], 
                                              self._params.X_symbols[i], 
                                              self._params.zeta, 
                                              self._params.epsilon,
                                              self._params.m,
                                              self._params.K,
                                              self.lambda_mat,
                                              states_cardinality=len(self._params.X_symbols))
                else:
                    B[i, j] = cond_prob_y_given_x(self._params.Y_symbols[j], 
                                                  self._params.X_symbols[i], 
                                                  self._params.zeta, 
                                                  self._params.epsilon,
                                                  self._params.m,
                                                  self._params.K,
                                                  self._params.alpha,
                                                  self._params.R)

        return B
    
    def _generate_lamda_matrix(self, noise_distribution="uniform") -> np.ndarray:
        shape = (len(self._params.X_symbols), len(self._params.X_symbols), self._params.K)
        out = np.empty(shape, dtype=float)
        for d in range(shape[2]):
            lam_val = lam(d, self._params.alpha, self._params.R_unit)
            if noise_distribution == "uniform":
                other_lam_val = (1 - lam_val) / (shape[1]-1)
            else:
                raise NotImplementedError("The method only supports uniform distribution for the other lambdas")
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i == j:
                        out[i, j, d] = lam_val
                    else:
                        out[i, j, d] = other_lam_val

        return out
    
    def _build_tranmission_matrix(self):
        shape = (len(self._params.X_symbols), len(self._params.X_symbols))
        A = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i == j:
                    A[i, j] = 1 - 2 * self._params.q
                elif j == i + 1:
                    A[i, j] = self._params.q
                elif j == i - 1:
                    A[i, j] = self._params.q
                else:
                    continue
        A[0, 0] = 1 - self._params.q
        A[-1, -1] = 1 - self._params.q
        return A
    

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