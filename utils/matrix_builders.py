from .hmm_funcs import lam

import numpy as np
from numba import jit

@jit
def generate_lambda_matrix(num_x_symbols:int, K:int, alpha:float, R_unit:float, noise_distribution="uniform") -> np.ndarray:
    """
    Compute the matrix for lambda values, Third dimension is the zone index.
    First dimension is the state of the source, Second dimension is the received symbol.
    """
    shape = (num_x_symbols, num_x_symbols, K)
    out = np.empty(shape, dtype=float)
    for d in range(shape[2]):
        lam_val = lam(d, alpha, R_unit)
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

