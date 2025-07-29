from .hmm_funcs import lam

import numpy as np
from numba import jit
from typing import List

def generate_lambda_matrix(num_x_symbols:int, K:int, alpha:float, R_unit:float, noise_distribution="uniform", base_sigma:float=1) -> np.ndarray:
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
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if i == j:
                        out[i, j, d] = lam_val
                    else:
                        out[i, j, d] = other_lam_val
        elif noise_distribution == "gaussian":
            sigma = _compute_sigma(base_sigma, d)
            domain = np.arange(0, num_x_symbols)
            # create rows and fill the matrix
            for i in range(shape[0]):
                out[i, :, d] = _gaussian_split_with_center_non_symmetric(domain, center_value=i, sigma=sigma, lambd=lam_val)
        elif noise_distribution == "diagonal":
            out[:, :, d] = np.eye(shape[0], shape[1])
        elif noise_distribution == "partial_uniform":
            for i in range(shape[0]):
                out[i, :, d] = distribute_probability(num_x_symbols, i, lam_val, base_sigma)
        else:
            raise NotImplementedError("The method only supports uniform distribution for the other lambdas")
    return out

def _gaussian_split_with_center_non_symmetric(domain:np.ndarray, center_value:int, sigma:float, lambd:float) -> np.ndarray:
    # Find the index of the center value in the domain
    center_index = np.where(domain == center_value)[0][0]
    
    # Calculate Gaussian values for all points relative to the chosen center
    distances = domain - domain[center_index]
    gaussian_vals = np.exp(- (distances**2) / (2 * sigma**2))

    # Remove center point for normalization of the rest
    gaussian_vals_without_center = np.delete(gaussian_vals, center_index)
    
    # Normalize the rest so sum to 1
    gaussian_rest_norm = gaussian_vals_without_center / gaussian_vals_without_center.sum()

    # Allocate probabilities
    probs = np.zeros_like(domain, dtype=float)
    probs[center_index] = lambd
    # Assign the rest with scaled gaussian values
    rest_indices = np.arange(len(domain)) != center_index
    probs[rest_indices] = (1 - lambd) * gaussian_rest_norm
    
    return probs

def _compute_sigma(base_sigma:float, zone_idx:int) -> np.ndarray:
    return base_sigma

def distribute_probability(num_symbols:int, central_index:int, lambda_val:float, num_neighbors:int):
    """
    Distribute the remaining probability (lambda_val) among neighboring symbols in a discrete domain.

    Parameters:
    num_symbols (int): Number of symbols in the domain.
    central_index (int): Index of the central symbol.
    central_prob (float): Probability of the central symbol.
    num_neighbors (int): Number of neighboring symbols to distribute the probability across.

    Returns:
    list: A list of probabilities corresponding to the symbols in the domain.
    """
    # Ensure the total probability does not exceed 1
    if lambda_val > 1:
        raise ValueError("Central probability must not exceed 1")

    # Calculate the remaining probability (lambda_val)
    rem_prob = 1 - lambda_val

    # Initialize the probabilities array with zeros
    probabilities = np.zeros(num_symbols)

    # Assign the central symbol's probability
    probabilities[central_index] = lambda_val

    # Get the neighboring indices (evenly distributed)
    left = central_index - 1
    right = central_index + 1

    # Start by distributing the probability to the neighbors
    neighbors = []
    for _ in range(num_neighbors):
        if left >= 0:
            neighbors.append(left)
            left -= 1
        if right < num_symbols:
            neighbors.append(right)
            right += 1

    # Split the remaining probability (lambda_val) evenly among neighbors
    prob_per_neighbor = rem_prob / len(neighbors)
    
    # Assign the probabilities to the neighboring symbols
    for neighbor_index in neighbors:
        probabilities[neighbor_index] = prob_per_neighbor
    
    return probabilities
